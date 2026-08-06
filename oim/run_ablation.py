r"""One-axis-at-a-time ablation sweep for MPPI vs. ADMM on xArm6.

Not `oim/run_launch.py`: that sweeps the cartesian product of every axis it
is given, which is the wrong shape for "vary one knob around a fixed
baseline, on all 5 scenes, then move to the next knob" -- crossing every
axis with every other would be enormous and mostly redundant. This
enumerates the union of six one-axis sweeps instead, reusing
`run_launch.py`'s GPU-memory wait between cells since a cell here is the
same kind of subprocess call.

    uv run python -m oim.run_ablation
    uv run python -m oim.run_ablation --dry-run

Writes one run file + one summary PNG per cell to the usual
`oim/results/runs/` and `oim/recordings/` (never `--record`: a video per
cell would dwarf the sweep). A manifest recording which axis/value produced
each run file goes to `oim/results/ablations/ablation_<timestamp>.json`,
since a run file's own hyperparameters do not say *why* that value was
chosen over the baseline for that cell.
"""

import argparse
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List

from oim import ROOT
from oim.run_launch import _await_gpu, script_path

SCENES = [
    "open_table",
    "single_obstacle",
    "shelf_gap",
    "ycb_clutter",
    "icra_sign",
]

# Shared across every cell unless a cell's own axis overrides one entry.
BASELINE = dict(horizon=35, samples=96, n_admm=8, consensus_alpha=0.2)
SEED = 5
STEPS = 1000
# ycb_clutter alone runs longer: it is consistently the slowest scene to
# converge (dense contact, closest to a real cluttered pushing problem),
# so 1000 steps for it would cut off before the trend visible in the other
# scenes plays out.
STEPS_OVERRIDE = {"ycb_clutter": 1500}

# (axis name, algorithms it applies to, [values to try]). Each axis's own
# baseline value is included as one of its points, so every sweep's own
# data covers the shared baseline too -- no separate baseline-only cell.
AXES = [
    ("horizon", ["mppi", "admm"], [25, 35, 50, 64]),
    ("samples", ["mppi", "admm"], [64, 96, 128, 256]),
    ("n_admm", ["admm"], [8, 16, 32]),
    ("consensus_alpha", ["admm"], [0.1, 0.2, 0.5]),
    # None = paper's single scalar (rho for all three wrench dims); 10/20
    # split the torque penalty out from the force penalty (rho stays 10),
    # a gradual step up from no split. rho_torque=10 happens to equal the
    # baseline's own rho, so it will score identically to None -- not a
    # bug, just what asking for that exact value does when the baseline
    # force rho is also 10.
    ("rho_torque", ["admm"], [None, 10.0, 20.0]),
    # ADMM's own sub-optimizers stay at 1 inner iteration (Algorithm 4,
    # and measured worse at 3/6 earlier this session) -- this axis is
    # flat MPPI only, the "vanilla, more inner iterations" side of the
    # iterations-vs-n_admm comparison.
    ("iterations", ["mppi"], [1, 4, 8]),
]


def build_cells() -> List[Dict[str, Any]]:
    """Every (scene, algorithm, axis, value) cell the sweep runs.

    Returns:
        One dict per cell: `scene`, `algorithm`, `axis`, `value`, and
        `flags` (the full flag dict for that cell, baseline plus the one
        overridden axis).
    """
    cells = []
    for axis, algorithms, values in AXES:
        for algorithm in algorithms:
            for value in values:
                flags = dict(BASELINE)
                if algorithm == "admm":
                    flags.pop("iterations", None)
                else:
                    flags = {
                        k: v
                        for k, v in flags.items()
                        if k in ("horizon", "samples")
                    }
                if axis == "rho_torque":
                    flags.pop("rho_torque", None)
                    if value is not None:
                        flags["rho_torque"] = value
                else:
                    flags[axis] = value
                for scene in SCENES:
                    cells.append(
                        dict(
                            scene=scene,
                            algorithm=algorithm,
                            axis=axis,
                            value=value,
                            flags=dict(flags),
                        )
                    )
    return cells


# Flags the top-level parser owns (before the algorithm subcommand);
# everything else in a cell's `flags` belongs to the subcommand.
_TOP_LEVEL_FLAGS = {"horizon", "samples"}


def build_command(cell: Dict[str, Any]) -> List[str]:
    """One cell's argv, in the same shape `run_launch.build_command` uses."""
    steps = STEPS_OVERRIDE.get(cell["scene"], STEPS)
    pre = ["--robot", "xarm6", "--warp"]
    post = ["--headless", "--seed", str(SEED), "--steps", str(steps)]
    for key, value in cell["flags"].items():
        flag = "--" + key.replace("_", "-")
        target = pre if key in _TOP_LEVEL_FLAGS else post
        target += [flag, str(value)]
    return [
        sys.executable,
        script_path(cell["scene"]),
        *pre,
        cell["algorithm"],
        *post,
    ]


def _label(cell: Dict[str, Any]) -> str:
    return (
        f"{cell['scene']}/{cell['algorithm']} "
        f"{cell['axis']}={cell['value']}"
    )


def _run_cell(cell: Dict[str, Any], timeout: float) -> Dict[str, Any]:
    """Run one cell, returning a manifest record.

    A run file's own name is timestamp-based, so the record captures the
    exact command instead of trying to guess which file it produced --
    `oim/run_eval.py` groups by the hyperparameters inside the file itself,
    and this manifest is only for "which axis was this cell testing".
    """
    cmd = build_command(cell)
    print(f"$ {' '.join(cmd)}")
    t0 = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=os.path.dirname(ROOT),
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        status = "ok" if proc.returncode == 0 else "failed"
        tail = "\n".join(proc.stdout.splitlines()[-5:])
        err_tail = (
            "\n".join(proc.stderr.splitlines()[-15:])
            if status == "failed"
            else ""
        )
    except subprocess.TimeoutExpired:
        status, tail, err_tail = "timeout", "", ""
    elapsed = time.time() - t0
    print(f"  {status} in {elapsed / 60:.1f} min")
    if status == "failed":
        print(f"  stderr tail:\n{err_tail}")
    return dict(
        cell,
        status=status,
        elapsed_s=elapsed,
        stdout_tail=tail,
        stderr_tail=err_tail,
    )


def main() -> None:
    """Parse arguments, run every cell, save the manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dry-run", action="store_true")
    p.add_argument(
        "--cell-timeout",
        type=float,
        default=3600.0,
        help="Seconds before a cell is killed and marked timed out.",
    )
    p.add_argument(
        "--gpu-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for GPU memory before running anyway.",
    )
    args = p.parse_args()

    cells = build_cells()
    print(f"{len(cells)} cells, steps={STEPS}, seed={SEED}")
    if args.dry_run:
        for cell in cells:
            print(f"  {_label(cell)}: {' '.join(build_command(cell))}")
        return

    records = []
    for i, cell in enumerate(cells):
        print(f"\n[{i + 1}/{len(cells)}] {_label(cell)}")
        _await_gpu(timeout=args.gpu_timeout)
        records.append(_run_cell(cell, args.cell_timeout))

    failed = [r for r in records if r["status"] != "ok"]
    print(f"\n{len(records)} cells, {len(failed)} failed/timed out")
    for r in failed:
        print(f"  {r['status'].upper()}: {_label(r)}")

    manifest_dir = os.path.join(ROOT, "results", "ablations")
    os.makedirs(manifest_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(manifest_dir, f"ablation_{stamp}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "baseline": BASELINE,
                "seed": SEED,
                "steps": STEPS,
                "steps_override": STEPS_OVERRIDE,
                "runs": records,
            },
            f,
            indent=2,
        )
    print(f"saved manifest to {path}")


if __name__ == "__main__":
    main()
