"""Utilities for persisting per-run hyperparameters and outcomes.

So a run's `rho`/`horizon`/`n_admm`/etc. and whether it actually reached the
goal are traceable later, alongside the recording -- not just visible in the
terminal output at the time.
"""

import json
import os
from typing import Any, Dict


def save_run_results(
    output_dir: str,
    name: str,
    hyperparameters: Dict[str, Any],
    log: Dict[str, Any],
) -> str:
    """Save a run's hyperparameters and outcome to a JSON file.

    Works for both `oim.sim2d.run.run_2d`'s and `oim.sim3d.run.run_3d_admm`'s
    log dicts, since both share the same diagnostic keys.

    Args:
        output_dir: Directory to save into (created if missing).
        name: Base filename, no extension -- pass the same name used for
            the recording/plot, so the two files pair up.
        hyperparameters: Whatever was used to build the task/controller
            (horizon, rho, n_admm, num_samples, seed, ...).
        log: The run's log dict. Only the per-step scalar diagnostics and
            the outcome are kept, not the full state trajectories (already
            captured visually in the plot/recording).

    Returns:
        The path the results were saved to.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name}.json")
    results = {
        "hyperparameters": hyperparameters,
        "reached": bool(log["reached"]),
        "steps_run": len(log["primal_residual"]),
        "primal_residual": list(log["primal_residual"]),
        "dual_residual": list(log["dual_residual"]),
        "rho": list(log["rho"]),
        "pos_err": list(log["pos_err"]),
        "theta_err": list(log["theta_err"]),
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved results to {path}")
    return path
