r"""Sweep driver: run `examples/pusht.py` once per parameter combination.

A benchmark or ablation is a cartesian product -- tasks x algorithms x
horizons x sampler budgets x seeds -- and this runs it. It holds no
planning code of its own: each cell is a subprocess invocation of
`examples/pusht.py`, so a cell is exactly a command you could have typed,
and the launcher prints that command before running it.

    # the whole product in oim/configs/run_launch_config.yaml
    uv run python -m oim.run_launch

    # see what would run, without running it
    uv run python -m oim.run_launch --dry-run

    # narrow a sweep without editing the config
    uv run python -m oim.run_launch --only algorithm=admm --only horizon=15

Scoring is deliberately elsewhere: this writes run files, and
`oim/run_eval.py` turns them into tables. A sweep is expensive and a metric
is cheap, so they must not share a lifetime.

Subprocess isolation is the point of not importing the runners directly: a
crashed or OOM cell loses one combination instead of the sweep, and each
run starts with a clean JAX allocator. The cost is a recompile per cell,
which is unavoidable anyway whenever the horizon or sample count changes
the traced shapes.
"""

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import yaml

from oim import ROOT

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
PUSHT = os.path.join(os.path.dirname(ROOT), "examples", "pusht.py")

# Sweep axes, in nesting order: earlier axes vary slowest, so every cell of
# one task finishes before the next task starts.
_AXES = (
    "task",
    "algorithm",
    "robot_opt",
    "object_opt",
    "horizon",
    "samples",
    "n_admm",
    "seed",
)


def _flag_spec() -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Ask `examples/pusht.py` which flags it takes, and where.

    Its parser splits on the algorithm name -- world and scene flags before
    it, solver and run flags after -- and mixes store_true switches with
    valued options. Rather than restate that here (a copy that silently
    rots the moment a flag moves), the classification is read off the real
    parser: a wrong guess would otherwise surface as every cell of a long
    sweep failing identically.

    Returns:
        `(top_level, per_algorithm)`, each mapping a dest name to whether
        the flag takes a value (False means a bare switch).
    """
    # Importing pusht.py builds module-level `jnp` arrays (`GOAL`,
    # `CLUTTER_OBSTACLES`), which initializes JAX's GPU backend and claims
    # ~75% of the device -- in *this* process, which then holds it for the
    # whole sweep and starves every cell of it. The launcher only reads a
    # parser, so pin it to CPU first.
    #
    # `jax.config`, not `JAX_PLATFORMS`: the environment variable is read
    # when `jax` is first imported, and `oim/__init__.py` has already done
    # that by the time this runs. Setting it here would silently do
    # nothing. Children are unaffected either way -- they are separate
    # processes with their own JAX.
    jax.config.update("jax_platforms", "cpu")
    spec = importlib.util.spec_from_file_location("_pusht_cli", PUSHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module._build_parser()

    top: Dict[str, bool] = {}
    sub: Dict[str, bool] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                for a in subparser._actions:
                    if a.dest != "help":
                        sub[a.dest] = a.nargs != 0
        elif action.dest not in ("help",):
            top[action.dest] = action.nargs != 0
    return top, sub


def load_config(path: str) -> Dict[str, Any]:
    """Load a sweep config.

    Args:
        path: Path to the YAML, or a bare name resolved in `oim/configs/`.

    Returns:
        The parsed config, with `sweep` and `fixed` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, path)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def expand(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the sweep axes into one dict per combination.

    Cells that cannot exist are dropped rather than left to fail at
    runtime: 2D has no flat baselines, and a flat baseline has no ADMM
    blocks, so sweeping `robot_opt` would otherwise re-run the identical
    baseline once per pair.

    Args:
        sweep: The config's `sweep` block, `{axis: [values]}`.

    Returns:
        One dict per valid combination, in nesting order.
    """
    axes = [a for a in _AXES if sweep.get(a)]
    combos = []
    seen = set()
    for values in itertools.product(*(sweep[a] for a in axes)):
        cell = dict(zip(axes, values, strict=True))
        task = cell.get("task", {})
        world = task.get("world", "3d")
        algorithm = cell.get("algorithm", "admm")

        if world == "2d" and algorithm != "admm":
            continue  # PushT2D implements only ConsensusTask
        if algorithm != "admm":
            # Flat baselines have no blocks; collapse the duplicates.
            cell = {
                k: v
                for k, v in cell.items()
                if k not in ("robot_opt", "object_opt", "n_admm")
            }
        key = json.dumps(cell, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        combos.append(cell)
    return combos


def build_command(
    cell: Dict[str, Any],
    fixed: Dict[str, Any],
    spec: Optional[Tuple[Dict[str, bool], Dict[str, bool]]] = None,
) -> List[str]:
    """Turn one sweep cell into an `examples/pusht.py` command line.

    Args:
        cell: One combination from `expand`.
        fixed: The config's `fixed` block, applied to every cell.
        spec: `_flag_spec()` output; read once and reused across a sweep.

    Returns:
        The argv list, algorithm name in its required position.

    Raises:
        ValueError: If a config key is not a flag `examples/pusht.py`
            accepts -- caught here rather than as an argparse error
            repeated once per cell.
    """
    top, sub = spec if spec is not None else _flag_spec()
    task = dict(cell.get("task", {}))
    algorithm = cell.get("algorithm", "admm")
    settings = {
        **fixed,
        **{k: v for k, v in cell.items() if k not in ("task", "algorithm")},
        **{k: v for k, v in task.items()},
    }

    pre: List[str] = []
    post: List[str] = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if key in top:
            target, takes_value = pre, top[key]
        elif key in sub:
            target, takes_value = post, sub[key]
        else:
            raise ValueError(
                f"'{key}' is not an examples/pusht.py flag; check the sweep "
                f"config. Known: {sorted(set(top) | set(sub))}"
            )
        if takes_value:
            target += [flag, str(value)]
        elif value:
            target.append(flag)

    return [sys.executable, PUSHT, *pre, algorithm, *post]


def _gpu_memory() -> Optional[Tuple[int, int]]:
    """`(free, total)` MiB on GPU 0, or None if there is no `nvidia-smi`.

    Returns:
        The reading, or None on a machine without an NVIDIA GPU -- the
        sweep must still run there, just without the memory barrier.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    free, total = out.stdout.strip().split("\n")[0].split(",")
    return int(free), int(total)


def _await_gpu(
    min_free_mib: int = 0, timeout: float = 120.0, poll: float = 0.5
) -> Optional[int]:
    """Wait for the previous cell's GPU memory to come back.

    A cell is its own process, so the driver reclaims on exit -- measured
    at well under a second, since the process is only reaped after CUDA
    teardown. This barrier exists for the cases where that reasoning does
    not hold: a cell that crashed rather than exited, a viewer or notebook
    someone left on the card, or a second sweep started by mistake. The
    xArm6 ADMM cell peaks near 12 GB of 16, so a few hundred stray MiB is
    the difference between a sweep and a column of Warp OOMs.

    Args:
        min_free_mib: Require at least this much free before returning. 0
            waits only for the reading to settle.
        timeout: Give up waiting after this long and run anyway -- an
            unmet requirement is reported by the cell that fails, which is
            more informative than the launcher refusing to start.
        poll: Seconds between readings.

    Returns:
        Free MiB at the moment of returning, or None without a GPU.
    """
    reading = _gpu_memory()
    if reading is None:
        return None

    deadline = time.time() + timeout
    free, _ = reading
    previous = -1
    while time.time() < deadline:
        # Settled (two equal readings) and enough headroom: go.
        if free == previous and free >= min_free_mib:
            return free
        previous = free
        time.sleep(poll)
        free = _gpu_memory()[0]

    print(
        f"  warning: only {free} MiB free after waiting {timeout:.0f}s "
        f"(wanted {min_free_mib} MiB); running anyway"
    )
    return free


def _label(cell: Dict[str, Any]) -> str:
    """A short human-readable name for one cell, for progress output."""
    task = cell.get("task", {})
    parts = [task.get("world", "3d")]
    parts += [str(task[k]) for k in ("robot", "scene", "env") if k in task]
    parts.append(str(cell.get("algorithm", "admm")))
    parts += [
        f"{k}={cell[k]}"
        for k in ("horizon", "samples", "n_admm", "seed")
        if k in cell
    ]
    return " ".join(parts)


def run_sweep(
    combos: Sequence[Dict[str, Any]],
    fixed: Dict[str, Any],
    dry_run: bool = False,
    keep_going: bool = True,
    min_free_mib: int = 0,
    gpu_timeout: float = 120.0,
) -> List[Dict[str, Any]]:
    """Run every combination, reporting progress and collecting outcomes.

    Args:
        combos: Cells from `expand`.
        fixed: The config's `fixed` block.
        dry_run: Print the commands without running them.
        keep_going: Continue after a cell fails. On by default -- a sweep
            is long and one bad combination should not discard the rest.
        min_free_mib: GPU memory required before a cell starts; see
            `_await_gpu`.
        gpu_timeout: How long to wait for it.

    Returns:
        One record per cell: its settings, command, exit status, duration,
        and the free GPU memory it started with.
    """
    records = []
    spec = _flag_spec()
    for i, cell in enumerate(combos, 1):
        cmd = build_command(cell, fixed, spec)
        print(f"\n[{i}/{len(combos)}] {_label(cell)}")
        print("  " + " ".join(cmd))
        if dry_run:
            records.append({"cell": cell, "command": cmd, "status": "skipped"})
            continue

        # Between cells, not only after: this also catches memory held by
        # something that was not part of this sweep.
        free = _await_gpu(min_free_mib, gpu_timeout)
        if free is not None:
            print(f"  {free} MiB free")

        t0 = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        print(f"  {'ok' if ok else 'FAILED'} in {elapsed:.1f}s")
        records.append(
            {
                "cell": cell,
                "command": cmd,
                "status": "ok" if ok else "failed",
                "returncode": result.returncode,
                "seconds": elapsed,
                "gpu_free_mib_at_start": free,
            }
        )
        if not ok and not keep_going:
            print("  stopping (--stop-on-error)")
            break
    return records


def _parse_only(only: Sequence[str]) -> Dict[str, str]:
    """Parse repeated `--only key=value` filters."""
    return dict(o.split("=", 1) for o in only)


def _parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` into typed values.

    Values go through the YAML loader, so `true`, `50` and `0.1` arrive as
    a bool, an int and a float rather than as strings -- `warp=false` has
    to be falsy, or switching a `fixed:` entry off from the command line
    would silently switch it on.

    Args:
        overrides: Raw `key=value` strings.

    Returns:
        The parsed overrides.

    Raises:
        ValueError: If an entry has no `=`.
    """
    parsed: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(value)
    return parsed


def _check_fixed(fixed: Dict[str, Any]) -> None:
    """Fail early if a `fixed:` key is not an `examples/pusht.py` flag.

    Otherwise a typo surfaces as every cell of a long sweep failing
    identically, minutes apart, with an argparse error buried in each
    subprocess's output.

    Args:
        fixed: The merged `fixed:` block and CLI overrides.

    Raises:
        ValueError: If any key is unknown.
    """
    top, sub = _flag_spec()
    unknown = sorted(set(fixed) - set(top) - set(sub))
    if unknown:
        raise ValueError(
            f"not examples/pusht.py flags: {unknown}. "
            f"Known: {sorted(set(top) | set(sub))}"
        )


def _apply_only(
    combos: List[Dict[str, Any]], only: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Keep cells matching every filter, comparing as strings."""
    kept = []
    for cell in combos:
        flat = {**cell.get("task", {}), **cell}
        if all(str(flat.get(k)) == v for k, v in only.items()):
            kept.append(cell)
    return kept


def main() -> None:
    """Parse arguments, expand the sweep, run it, save a manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="run_launch_config",
        help="Sweep config: a path, or a name under oim/configs/.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep only cells matching this; repeatable.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and the cell count, run nothing.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep on the first failing cell.",
    )
    p.add_argument(
        "--manifest-dir",
        default=os.path.join(ROOT, "results", "sweeps"),
        help="Where to write the record of what ran.",
    )
    p.add_argument(
        "--min-free-mib",
        type=int,
        default=0,
        help="Wait for at least this much free GPU memory before each cell. "
        "0 waits only for the reading to settle. The xArm6 ADMM cell peaks "
        "near 12000.",
    )
    p.add_argument(
        "--gpu-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for GPU memory before running anyway.",
    )
    p.add_argument(
        "--warp",
        action="store_true",
        help="Run every cell on the MuJoCo Warp backend "
        "(shorthand for --set warp=true).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override a `fixed:` entry for this sweep only; repeatable. "
        "Any examples/pusht.py flag works, e.g. --set steps=50 "
        "--set record=false.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    combos = expand(cfg.get("sweep", {}))
    if args.only:
        combos = _apply_only(combos, _parse_only(args.only))

    fixed = dict(cfg.get("fixed", {}))
    if args.warp:
        fixed["warp"] = True
    try:
        fixed.update(_parse_overrides(args.overrides))
        _check_fixed(fixed)
    except ValueError as e:
        p.error(str(e))

    if not combos:
        print("no cells to run (check the sweep config and --only filters)")
        return

    print(f"{len(combos)} cells from {args.config}")
    t0 = time.time()
    records = run_sweep(
        combos,
        fixed,
        dry_run=args.dry_run,
        keep_going=not args.stop_on_error,
        min_free_mib=args.min_free_mib,
        gpu_timeout=args.gpu_timeout,
    )
    total = time.time() - t0

    failed = [r for r in records if r["status"] == "failed"]
    print(
        f"\n{len(records)} cells in {total / 60:.1f} min, {len(failed)} failed"
    )
    for r in failed:
        print(f"  FAILED: {_label(r['cell'])}")
    if args.dry_run:
        return

    os.makedirs(args.manifest_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.manifest_dir, f"sweep_{stamp}.json")
    with open(path, "w") as f:
        json.dump(
            {"config": args.config, "fixed": fixed, "runs": records},
            f,
            indent=2,
        )
    print(f"saved manifest to {path}")
    print("\nnext: uv run python -m oim.run_eval")


if __name__ == "__main__":
    main()
