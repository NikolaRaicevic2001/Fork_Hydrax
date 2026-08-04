r"""Compute metrics over saved runs -- without re-running any experiment.

`examples/pusht.py` produces one run file per run; `oim/run_launch.py`
produces many. This reads them, derives every metric on the fly via
`oim.utils.metrics`, groups the runs into comparable cells, and emits the
table that goes in the paper.

    # every run under oim/results/runs, grouped automatically
    uv run python -m oim.run_eval

    # one sweep, grouped explicitly, as a LaTeX-ready markdown table
    uv run python -m oim.run_eval --runs-dir oim/results/runs \\
        --group-by task algorithm horizon --format markdown

    # summarise a chosen set of tasks / algorithms / horizons
    uv run python -m oim.run_eval \
        --filter task=pusht3d_xarm6,pusht3d_point \
        --filter algorithm=admm,mppi --filter horizon=5,15,25 \
        --group-by task algorithm horizon

    # re-score old runs against a stricter success tolerance
    uv run python -m oim.run_eval --pos-tol 0.02 --theta-tol 0.02

Nothing here touches the simulator, JAX or MuJoCo: changing a metric, a
tolerance, or how results are grouped costs a second, not a GPU-week. That
is the entire reason running and scoring are separate programs.
"""

import argparse
import glob
import json
import os
from collections import defaultdict
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from oim import ROOT
from oim.utils.metrics import aggregate_metrics, trial_metrics
from oim.utils.results import RunName, load_run

# Group keys always used, if present, before any auto-detected ones. `seed`
# is deliberately absent: seeds are the *trials within* a cell, not a cell.
_BASE_KEYS = ("world", "task", "algorithm", "robot_opt", "object_opt")

# Columns of the emitted table, as (header, metric key, format).
_COLUMNS: Tuple[Tuple[str, str, str], ...] = (
    ("n", "n_trials", "d"),
    ("SR", "success_rate", ".2f"),
    ("eps_d", "pos_err_mean", ".4f"),
    ("+/-", "pos_err_std", ".4f"),
    ("eps_d^s", "pos_err_mean_success", ".4f"),
    ("theta", "theta_err_mean", ".4f"),
    ("+/-", "theta_err_std", ".4f"),
    ("T", "mean_execution_time", ".2f"),
    ("f (Hz)", "mean_frequency_hz", ".2f"),
)


def _run_fields(run: Dict[str, Any]) -> Dict[str, Any]:
    """Flatten a run's identity and settings into one lookup table."""
    fields = dict(run.get("run", {}))
    fields.update(run.get("hyperparameters", {}))
    return fields


def _auto_group_keys(runs: List[Dict[str, Any]]) -> List[str]:
    """Group by identity, plus any setting that actually varies.

    A sweep over horizon should become a horizon column without being
    asked; a setting held fixed across every run should not, since a column
    of one repeated value is noise. `seed` is excluded because varying it
    is what makes trials rather than cells.
    """
    fields = [_run_fields(r) for r in runs]
    keys = [k for k in _BASE_KEYS if any(k in f for f in fields)]
    varying = set()
    for key in {k for f in fields for k in f}:
        if key in keys or key == "seed":
            continue
        seen = {json.dumps(f.get(key), sort_keys=True) for f in fields}
        if len(seen) > 1:
            varying.add(key)
    return keys + sorted(varying)


def parse_filters(items: Sequence[str]) -> Dict[str, Set[str]]:
    """Parse `--filter key=a,b` and repeated flags into `{key: {values}}`.

    Values accumulate rather than overwrite, so `--filter task=a --filter
    task=b` and `--filter task=a,b` mean the same thing: *any of these*.
    Overwriting would silently evaluate a different set of runs than the
    one asked for.

    Args:
        items: Raw `key=value[,value...]` strings.

    Returns:
        One set of accepted values per field.

    Raises:
        ValueError: If an entry has no `=`.
    """
    parsed: Dict[str, Set[str]] = defaultdict(set)
    for item in items:
        if "=" not in item:
            raise ValueError(f"--filter expects KEY=VALUE, got '{item}'")
        key, values = item.split("=", 1)
        parsed[key.strip()].update(v.strip() for v in values.split(","))
    return dict(parsed)


def _matches(run: Dict[str, Any], filters: Dict[str, Set[str]]) -> bool:
    """Whether a run is any of the accepted values for every filtered field.

    Within a field the values are OR-ed, across fields AND-ed -- so
    `--filter task=a,b --filter algorithm=admm` reads as "either task, but
    only ADMM".
    """
    fields = _run_fields(run)
    return all(str(fields.get(k)) in values for k, values in filters.items())


def evaluate(
    runs: List[Dict[str, Any]],
    group_by: Sequence[str],
    shared_max_time: bool = True,
) -> Dict[str, Dict[str, Any]]:
    """Aggregate loaded runs into one metric set per group.

    Args:
        runs: Payloads from `oim.utils.results.load_run`.
        group_by: Field names forming each group's key.
        shared_max_time: Credit unsuccessful trials the slowest execution
            time *across the whole sweep* rather than within their own
            group. Keeps the T column comparable between methods: scored
            per group, a method that fails quickly and consistently would
            be credited its own short worst case and beat one that nearly
            succeeds.

    Returns:
        `{group_label: aggregate_metrics(...)}`, ordered by label.
    """
    grouped: Dict[Tuple, List[Dict[str, Any]]] = defaultdict(list)
    for run in runs:
        fields = _run_fields(run)
        grouped[tuple(fields.get(k) for k in group_by)].append(
            trial_metrics(run)
        )

    max_time = None
    if shared_max_time:
        max_time = max(
            t["execution_time"] for ts in grouped.values() for t in ts
        )

    return {
        "/".join(str(v) for v in key): aggregate_metrics(
            trials, max_time=max_time
        )
        for key, trials in sorted(grouped.items(), key=lambda kv: str(kv[0]))
    }


def format_table(
    summary: Dict[str, Dict[str, Any]],
    group_by: Sequence[str],
    markdown: bool = False,
) -> str:
    """Render the summary as a fixed-width or markdown table.

    Args:
        summary: `evaluate` output.
        group_by: The field names its labels were built from.
        markdown: Emit a markdown table instead of aligned plain text.

    Returns:
        The rendered table.
    """
    headers = ["/".join(group_by)] + [c[0] for c in _COLUMNS]
    rows = []
    for label, m in summary.items():
        row = [label]
        for _, key, fmt in _COLUMNS:
            value = m.get(key)
            row.append("-" if value is None else format(value, fmt))
        rows.append(row)

    if markdown:
        out = ["| " + " | ".join(headers) + " |"]
        out.append("| " + " | ".join("---" for _ in headers) + " |")
        out += ["| " + " | ".join(r) + " |" for r in rows]
        return "\n".join(out)

    widths = [
        max(len(headers[i]), *(len(r[i]) for r in rows))
        for i in range(len(headers))
    ]

    def _row(cells: Sequence[str]) -> str:
        pairs = zip(cells, widths, strict=True)
        return "  ".join(c.ljust(w) for c, w in pairs)

    line = _row(headers)
    return "\n".join([line, "-" * len(line)] + [_row(r) for r in rows])


def load_runs(
    runs_dir: str,
    filters: Optional[Dict[str, Set[str]]] = None,
    pos_tol: Optional[float] = None,
    theta_tol: Optional[float] = None,
) -> List[Dict[str, Any]]:
    """Load every run file under `runs_dir`, optionally filtered and re-scored.

    Args:
        runs_dir: Directory of `save_run` JSON files.
        filters: `{field: {accepted values}}` from `parse_filters`,
            compared as strings.
        pos_tol: Override the recorded positional success tolerance.
        theta_tol: Override the recorded angular one. Overriding either
            re-scores old runs under a new definition of success, which is
            the point of storing poses rather than a `reached` flag.

    Returns:
        The matching payloads.

    Raises:
        FileNotFoundError: If no run files match.
    """
    paths = sorted(glob.glob(os.path.join(runs_dir, "*.json")))
    loaded = [load_run(p) for p in paths]
    if filters:
        known = {k for r in loaded for k in _run_fields(r)}
        unknown = sorted(set(filters) - known)
        if unknown:
            raise ValueError(
                f"no run records the field(s) {unknown}. "
                f"Available: {sorted(known)}"
            )

    runs = []
    for run in loaded:
        if filters and not _matches(run, filters):
            continue
        if pos_tol is not None:
            run["hyperparameters"]["goal_pos_tol"] = pos_tol
        if theta_tol is not None:
            run["hyperparameters"]["goal_theta_tol"] = theta_tol
        runs.append(run)
    if not runs:
        raise FileNotFoundError(
            f"no run files matched in {runs_dir} "
            f"({len(paths)} present, {filters or 'no'} filters)"
        )
    return runs


def main() -> None:
    """Parse arguments, aggregate the runs, print and save the table."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--runs-dir",
        default=os.path.join(ROOT, "results", "runs"),
        help="Directory of run files written by examples/pusht.py.",
    )
    p.add_argument(
        "--group-by",
        nargs="+",
        default=None,
        help="Fields forming each table row. Default: identity fields plus "
        "any hyperparameter that varies across the loaded runs.",
    )
    p.add_argument(
        "--filter",
        action="append",
        default=[],
        metavar="KEY=VALUE[,VALUE...]",
        help="Keep only runs whose field is one of these values; "
        "repeatable, and comma-separated lists are accepted. Values of one "
        "field are OR-ed, different fields AND-ed.",
    )
    p.add_argument("--pos-tol", type=float, help="Re-score success with this.")
    p.add_argument("--theta-tol", type=float, help="Re-score success.")
    p.add_argument(
        "--format",
        choices=["text", "markdown"],
        default="text",
        help="Table style for stdout and the saved .md/.txt.",
    )
    p.add_argument(
        "--out-dir",
        default=os.path.join(ROOT, "results", "eval"),
        help="Where to write the summary JSON and table.",
    )
    p.add_argument("--no-save", action="store_true")
    args = p.parse_args()

    try:
        filters = parse_filters(args.filter)
        runs = load_runs(args.runs_dir, filters, args.pos_tol, args.theta_tol)
    except (ValueError, FileNotFoundError) as e:
        p.error(str(e))
    group_by = args.group_by or _auto_group_keys(runs)
    summary = evaluate(runs, group_by)

    table = format_table(summary, group_by, markdown=args.format == "markdown")
    print(f"\n{len(runs)} runs, grouped by {group_by}\n")
    print(table)

    if args.no_save:
        return
    os.makedirs(args.out_dir, exist_ok=True)
    name = RunName("eval")
    json_path = os.path.join(args.out_dir, f"{name()}.json")
    with open(json_path, "w") as f:
        json.dump(
            {
                "runs_dir": args.runs_dir,
                "n_runs": len(runs),
                "filters": filters,
                "group_by": list(group_by),
                "goal_pos_tol": args.pos_tol,
                "goal_theta_tol": args.theta_tol,
                "results": summary,
            },
            f,
            indent=2,
        )
    ext = "md" if args.format == "markdown" else "txt"
    table_path = os.path.join(args.out_dir, f"{name()}.{ext}")
    with open(table_path, "w") as f:
        f.write(table + "\n")
    print(f"\nsaved {json_path}\nsaved {table_path}")


if __name__ == "__main__":
    main()
