"""Aggregate metrics for comparing ADMM against the flat baselines.

Definitions follow the OI-MPPI paper (`Documentation/IROS2026.pdf`, Sec.
VI): success rate (SR), position error (eps_d, eps_d^s), control frequency
(f_bar), and total execution time (T). This module only aggregates numbers
already present in a `run_2d`/`run_3d_admm` log dict (or any log with the
same `pos_err`/`theta_err`/`reached`/`compute_time` keys) -- it does not run
anything itself; pair it with a driver that repeats trials under different
seeds/methods and collects their logs.

One difference from the paper, stated rather than hidden: `T` there is wall-
clock time on real hardware. Here `execution_time` is *simulated* task time
(`steps_run * dt`), since that is what is reproducible independent of
machine load; wall-clock planning cost is `mean_frequency_hz` instead,
computed from `compute_time` (present when the driver measured it).
"""

from typing import Any, Dict, List, Optional

import numpy as np


def trial_metrics(log: Dict[str, Any], dt: float) -> Dict[str, Any]:
    """Per-trial scalars extracted from one closed-loop run's log.

    Args:
        log: A `run_2d`/`run_3d_admm` (or equivalent) log dict.
        dt: The control period, for converting steps to simulated time.

    Returns:
        `reached`, `pos_err_mean` (over the whole trial, the paper's
        eps_d/eps_d^s before aggregation), `steps_run`, `execution_time`,
        and `mean_frequency_hz` (omitted if `log` has no `compute_time`).
    """
    pos_err = np.asarray(log["pos_err"])
    steps_run = int(len(pos_err))
    out: Dict[str, Any] = {
        "reached": bool(log["reached"]),
        "pos_err_mean": float(np.mean(pos_err)),
        "steps_run": steps_run,
        "execution_time": steps_run * dt,
    }
    compute_time = log.get("compute_time")
    if compute_time:
        out["mean_frequency_hz"] = float(1.0 / np.mean(compute_time))
    return out


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
        `n_trials`, `success_rate`, `pos_err_mean` (eps_d, over all trials),
        `pos_err_mean_success` (eps_d^s, over successful trials only --
        `None` if none succeeded), `mean_execution_time` (T), and
        `mean_frequency_hz` (f_bar, omitted if no trial has `compute_time`).
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

    result: Dict[str, Any] = {
        "n_trials": len(trials),
        "success_rate": len(successes) / len(trials),
        "pos_err_mean": float(np.mean([t["pos_err_mean"] for t in trials])),
        "pos_err_mean_success": (
            float(np.mean([t["pos_err_mean"] for t in successes]))
            if successes
            else None
        ),
        "mean_execution_time": float(np.mean(exec_times)),
    }
    if freqs:
        result["mean_frequency_hz"] = float(np.mean(freqs))
    return result
