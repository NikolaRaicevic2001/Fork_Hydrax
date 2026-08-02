"""Aggregate metrics for comparing ADMM against the flat baselines.

Definitions follow the OI-MPPI paper (`Documentation/IROS2026.pdf`, Sec.
VI): success rate (SR), position error (eps_d, eps_d^s), control frequency
(f_bar), and total execution time (T) -- plus orientation error (not in
the paper) and a std-dev alongside every mean. Aggregates numbers already
present in a `run_2d`/`run_3d_admm`/`run_3d_plain` log dict; does not run
anything.

`execution_time` is simulated task time (`steps_run * dt`), not the
paper's wall-clock hardware time -- reproducible independent of machine
load. Wall-clock planning cost is `mean_frequency_hz` instead, from
`compute_time` when the driver logged it.
"""

from typing import Any, Dict, List, Optional

import numpy as np


def trial_metrics(log: Dict[str, Any], dt: float) -> Dict[str, Any]:
    """Per-trial scalars extracted from one closed-loop run's log.

    Args:
        log: A `run_2d`/`run_3d_admm` (or equivalent) log dict.
        dt: The control period, for converting steps to simulated time.

    Returns:
        `reached`, `pos_err_mean`, `theta_err_mean` (each over the whole
        trial), `steps_run`, `execution_time`, and `mean_frequency_hz`
        (omitted if `log` has no `compute_time`).
    """
    pos_err = np.asarray(log["pos_err"])
    theta_err = np.asarray(log["theta_err"])
    steps_run = int(len(pos_err))
    out: Dict[str, Any] = {
        "reached": bool(log["reached"]),
        "pos_err_mean": float(np.mean(pos_err)),
        "theta_err_mean": float(np.mean(theta_err)),
        "steps_run": steps_run,
        "execution_time": steps_run * dt,
    }
    compute_time = log.get("compute_time")
    if compute_time:
        out["mean_frequency_hz"] = float(1.0 / np.mean(compute_time))
    return out


def _mean_std(values: List[float]) -> Dict[str, Optional[float]]:
    """Mean and population std of `values`, or `(None, None)` if empty."""
    if not values:
        return {"mean": None, "std": None}
    return {"mean": float(np.mean(values)), "std": float(np.std(values))}


def aggregate_metrics(
    logs: List[Dict[str, Any]], dt: float, max_time: Optional[float] = None
) -> Dict[str, Any]:
    """Aggregate several trials' logs into the paper's Table I/II columns.

    Args:
        logs: One log per trial (same task/method/hyperparameters, varying
            only initial condition and/or seed).
        dt: Control period, shared by every trial.
        max_time: Simulated execution time credited to a trial that never
            reached the goal, matching the paper's "record the max time
            allowed" convention. Defaults to the longest `execution_time`
            actually observed across `logs`.

    Returns:
        `n_trials`, `success_rate`; `pos_err_mean`/`_std` (eps_d, over all
        trials) and `pos_err_mean_success`/`_std_success` (eps_d^s, over
        successful trials only); the same four for `theta_err`;
        `mean_execution_time`/`std_execution_time` (T); and
        `mean_frequency_hz`/`std_frequency_hz` (f_bar, omitted if no trial
        has `compute_time`).
    """
    if not logs:
        raise ValueError("aggregate_metrics needs at least one trial")

    trials = [trial_metrics(log, dt) for log in logs]
    if max_time is None:
        max_time = max(t["execution_time"] for t in trials)

    successes = [t for t in trials if t["reached"]]
    exec_times = [
        t["execution_time"] if t["reached"] else max_time for t in trials
    ]
    freqs = [t["mean_frequency_hz"] for t in trials if "mean_frequency_hz" in t]
    pos_err = _mean_std([t["pos_err_mean"] for t in trials])
    pos_err_success = _mean_std([t["pos_err_mean"] for t in successes])
    theta_err = _mean_std([t["theta_err_mean"] for t in trials])
    theta_err_success = _mean_std([t["theta_err_mean"] for t in successes])
    exec_time = _mean_std(exec_times)

    result: Dict[str, Any] = {
        "n_trials": len(trials),
        "success_rate": len(successes) / len(trials),
        "pos_err_mean": pos_err["mean"],
        "pos_err_std": pos_err["std"],
        "pos_err_mean_success": pos_err_success["mean"],
        "pos_err_std_success": pos_err_success["std"],
        "theta_err_mean": theta_err["mean"],
        "theta_err_std": theta_err["std"],
        "theta_err_mean_success": theta_err_success["mean"],
        "theta_err_std_success": theta_err_success["std"],
        "mean_execution_time": exec_time["mean"],
        "std_execution_time": exec_time["std"],
    }
    if freqs:
        freq = _mean_std(freqs)
        result["mean_frequency_hz"] = freq["mean"]
        result["std_frequency_hz"] = freq["std"]
    return result
