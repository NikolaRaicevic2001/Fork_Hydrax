"""Multi-run ablation figures for `oim.run_eval --plot`.

Takes the already-aggregated step curves from `evaluate_step_curves` and
draws a task × metric grid. Deliberately separate from `plotting.py`, which
serves a single finished run's trajectory / cost panels.
"""

from typing import Any, Dict, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

# (column title, mean key, std key). Primal residual is ADMM-only; methods
# without it simply omit that line.
_METRICS: Tuple[Tuple[str, str, str], ...] = (
    (r"Cumulative SR $\uparrow$", "cum_success_mean", "cum_success_std"),
    (r"$\epsilon_d$ (m) $\downarrow$", "pos_err_mean", "pos_err_std"),
    (
        r"Primal residual $\downarrow$",
        "primal_residual_mean",
        "primal_residual_std",
    ),
)


def _subtitle(
    ablate: Sequence[str],
    filters: Optional[Mapping[str, Sequence[str]]],
) -> str:
    """One-line caption of what is compared vs pinned."""
    parts: List[str] = []
    if ablate:
        parts.append("ablate: " + ", ".join(ablate))
    if filters:
        pinned = ", ".join(
            f"{k}={','.join(v)}" for k, v in sorted(filters.items())
        )
        parts.append("filter: " + pinned)
    return "  |  ".join(parts)


def _draw_method(
    ax: Any,
    series: Dict[str, Any],
    method: str,
    color: Any,
    mean_key: str,
    std_key: str,
) -> None:
    """Plot one method's mean curve, with ±std band when n>1."""
    x = np.asarray(series["steps"])
    y = np.asarray(series[mean_key])
    ax.plot(x, y, color=color, label=method, lw=1.6)
    n = int(series.get("n_trials", 1))
    if n > 1 and std_key in series:
        s = np.asarray(series[std_key])
        ax.fill_between(x, y - s, y + s, color=color, alpha=0.18, lw=0)


def _collect_legend(axes: Any) -> Tuple[List[Any], List[str]]:
    """Union of legend entries across all axes (residual is ADMM-only)."""
    handles: List[Any] = []
    labels: List[str] = []
    seen: Set[str] = set()
    for ax in axes.ravel():
        h, lab = ax.get_legend_handles_labels()
        for handle, label in zip(h, lab, strict=True):
            if label not in seen:
                handles.append(handle)
                labels.append(label)
                seen.add(label)
    return handles, labels


def plot_step_curves(
    curves: Dict[str, Dict[str, Dict[str, Any]]],
    path: str,
    group_by: Sequence[str] = ("task",),
    ablate: Sequence[str] = (),
    filters: Optional[Mapping[str, Sequence[str]]] = None,
) -> str:
    """Write a rows=groups × cols=metrics figure comparing methods.

    Args:
        curves: `evaluate_step_curves` output (no Mean block expected).
        path: Destination PNG path.
        group_by: Fields the row labels were built from.
        ablate: Ablated fields, for the figure subtitle.
        filters: Pinned fields, for the figure subtitle.

    Returns:
        The path written.
    """
    # matplotlib backend must be set before pyplot loads; Agg keeps this
    # headless-safe on the same machines that run the sweep.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    # Imported lazily so this module never pulls `run_eval` at import time
    # (run_eval loads us only under `--plot`).
    from oim.run_eval import MEAN_LABEL, _strip_common_prefix  # noqa: PLC0415

    groups = [g for g in curves if g != MEAN_LABEL]
    if not groups:
        raise ValueError("plot_step_curves needs at least one row group")

    short, prefix = _strip_common_prefix(groups)
    labels = dict(zip(groups, short, strict=True))
    methods = sorted({m for g in groups for m in curves[g]})
    n_rows, n_cols = len(groups), len(_METRICS)

    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(4.2 * n_cols, 2.8 * n_rows),
        sharex="col",
        squeeze=False,
    )
    cmap = plt.get_cmap("tab10")
    colors = {m: cmap(i % 10) for i, m in enumerate(methods)}

    for r, group in enumerate(groups):
        for c, (title, mean_key, std_key) in enumerate(_METRICS):
            ax = axes[r, c]
            for method in methods:
                series = curves[group].get(method)
                if series is None or mean_key not in series:
                    continue
                _draw_method(
                    ax, series, method, colors[method], mean_key, std_key
                )
            if r == 0:
                ax.set_title(title)
            if c == 0:
                ax.set_ylabel(labels[group])
            if r == n_rows - 1:
                ax.set_xlabel("control step")
            ax.grid(True, alpha=0.3)
            if mean_key == "cum_success_mean":
                ax.set_ylim(-0.05, 1.05)

    handles, legend_labels = _collect_legend(axes)
    if handles:
        fig.legend(
            handles,
            legend_labels,
            loc="upper center",
            ncol=min(len(legend_labels), 4),
            frameon=False,
            bbox_to_anchor=(0.5, 1.02),
        )

    title = " / ".join(group_by) + " step curves"
    if prefix:
        title += f"  ({prefix.rstrip('_')} omitted)"
    sub = _subtitle(ablate, filters)
    fig.suptitle(title + (f"\n{sub}" if sub else ""), y=1.06 if sub else 1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return path
