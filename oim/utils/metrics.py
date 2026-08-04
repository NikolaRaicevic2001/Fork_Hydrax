"""Metrics derived from saved run files -- computed, never recorded.

Everything here is a function of what `oim.utils.results.save_run` already
wrote, so a new metric costs a re-read of the JSON rather than a re-run of
the experiment. That is the whole reason the two are separate: goal errors,
success, execution time and control frequency are *definitions*, and
definitions change.

Follows the OI-MPPI paper (`Documentation/IROS2026.pdf`, Sec. VI): success
rate (SR), position error (eps_d, eps_d^s), control frequency (f_bar) and
total execution time (T) -- plus orientation error, which the paper omits,
and a standard deviation beside every mean.

Deliberately numpy-only: `oim/run_eval.py` reads files and should not pay
for importing JAX or MuJoCo.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def _wrap(angle: np.ndarray) -> np.ndarray:
    """Wrap an angle to (-pi, pi]."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def goal_errors(run: Dict[str, Any]) -> Dict[str, np.ndarray]:
    """Per-step distance from the goal, over the executed steps.

    Uses `object_pose[1:]` -- the pose *after* each control step -- so the
    series aligns with the input and residual series rather than with the
    initial condition.

    Args:
        run: A payload from `oim.utils.results.load_run`.

    Returns:
        `pos_err` and `theta_err`, each of length `steps_run`.
    """
    goal = np.asarray(run["static"]["goal"], dtype=float)
    poses = np.asarray(run["dynamic"]["object_pose"], dtype=float)[1:]
    return {
        "pos_err": np.linalg.norm(poses[:, :2] - goal[:2], axis=1),
        "theta_err": np.abs(_wrap(poses[:, 2] - goal[2])),
    }


def trial_metrics(run: Dict[str, Any]) -> Dict[str, Any]:
    """Reduce one run file to the scalars an aggregate is built from.

    `reached` is re-derived from the final pose against the tolerances the
    run recorded, rather than trusted from a stored flag. That is exactly
    equivalent -- the closed loop exits the moment both tolerances are met,
    so a run that used all its steps never met them -- and it means a
    changed tolerance can be applied to old runs without re-running them.

    Args:
        run: A payload from `oim.utils.results.load_run`.

    Returns:
        `reached`, `pos_err_final`, `theta_err_final`, `pos_err_mean`,
        `theta_err_mean`, `steps_run`, `execution_time`, and
        `mean_frequency_hz` if the run recorded `compute_time`.
    """
    hp = run["hyperparameters"]
    err = goal_errors(run)
    pos_err, theta_err = err["pos_err"], err["theta_err"]
    steps_run = int(len(pos_err))

    pos_tol = float(hp["goal_pos_tol"])
    theta_tol = float(hp["goal_theta_tol"])
    reached = bool(
        steps_run > 0
        and pos_err[-1] < pos_tol
        and theta_err[-1] < theta_tol
    )

    out: Dict[str, Any] = {
        "reached": reached,
        "pos_err_final": float(pos_err[-1]) if steps_run else float("nan"),
        "theta_err_final": float(theta_err[-1]) if steps_run else float("nan"),
        "pos_err_mean": float(np.mean(pos_err)) if steps_run else float("nan"),
        "theta_err_mean": (
            float(np.mean(theta_err)) if steps_run else float("nan")
        ),
        "steps_run": steps_run,
        "execution_time": steps_run * float(hp["control_dt"]),
    }
    compute_time = run["dynamic"].get("compute_time")
    if compute_time:
        out["mean_frequency_hz"] = float(1.0 / np.mean(compute_time))
    return out


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    """Mean and population std of `values`, or `(None, None)` if empty."""
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate_metrics(
    trials: List[Dict[str, Any]], max_time: Optional[float] = None
) -> Dict[str, Any]:
    """Aggregate several trials of one method into the paper's columns.

    Args:
        trials: `trial_metrics` output, one per trial -- same task, method
            and hyperparameters, varying only the seed.
        max_time: Simulated execution time credited to a trial that never
            reached the goal. Defaults to the slowest trial in this group;
            pass the sweep-wide maximum to compare groups whose worst cases
            differ, since otherwise a method that fails fast scores better
            on T than one that nearly succeeds.

    Returns:
        `n_trials`, `success_rate`; `pos_err_mean`/`_std` over all trials
        and `pos_err_mean_success`/`_std_success` over successful ones; the
        same four for `theta_err`; `mean_`/`std_execution_time`; and
        `mean_`/`std_frequency_hz` when the trials recorded timings.

    Raises:
        ValueError: If `trials` is empty.
    """
    if not trials:
        raise ValueError("aggregate_metrics needs at least one trial")

    if max_time is None:
        max_time = max(t["execution_time"] for t in trials)

    successes = [t for t in trials if t["reached"]]
    exec_times = [
        t["execution_time"] if t["reached"] else max_time for t in trials
    ]
    freqs = [t["mean_frequency_hz"] for t in trials if "mean_frequency_hz" in t]

    pos_err = _mean_std([t["pos_err_mean"] for t in trials])
    pos_err_s = _mean_std([t["pos_err_mean"] for t in successes])
    theta_err = _mean_std([t["theta_err_mean"] for t in trials])
    theta_err_s = _mean_std([t["theta_err_mean"] for t in successes])
    exec_time = _mean_std(exec_times)

    result: Dict[str, Any] = {
        "n_trials": len(trials),
        "success_rate": len(successes) / len(trials),
        "pos_err_mean": pos_err["mean"],
        "pos_err_std": pos_err["std"],
        "pos_err_mean_success": pos_err_s["mean"],
        "pos_err_std_success": pos_err_s["std"],
        "theta_err_mean": theta_err["mean"],
        "theta_err_std": theta_err["std"],
        "theta_err_mean_success": theta_err_s["mean"],
        "theta_err_std_success": theta_err_s["std"],
        "mean_execution_time": exec_time["mean"],
        "std_execution_time": exec_time["std"],
    }
    if freqs:
        freq = _mean_std(freqs)
        result["mean_frequency_hz"] = freq["mean"]
        result["std_frequency_hz"] = freq["std"]
    return result
