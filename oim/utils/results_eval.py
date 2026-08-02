"""Persisting a multi-trial evaluation's hyperparameters and aggregate stats.

The `results.py` counterpart for a comparison across many trials/methods
rather than one run: `save_run_metrics`/`save_run_states` describe what one
run did; `save_eval_results` describes what `--trials` seeds of one or more
methods, aggregated via `oim.utils.metrics.aggregate_metrics`, did.
"""

import json
import os
from typing import Any, Dict

from oim.utils.results import RunName


def save_eval_results(
    output_dir: str,
    name: RunName,
    hyperparameters: Dict[str, Any],
    results: Dict[str, Any],
) -> str:
    """Save one eval run's settings and per-method aggregate metrics.

    Args:
        output_dir: Directory to save into (created if missing).
        name: The eval run's `RunName`.
        hyperparameters: Settings shared across every method compared
            (world, trials, steps, seed0, samples, horizon, ...).
        results: `{method_label: aggregate_metrics(...)}`, one entry per
            method compared (e.g. `"admm_mppi_mppi"`, `"mppi"`, `"ps"`).
            Already plain Python types -- `aggregate_metrics` casts every
            value to `float`/`None`, so no JSON conversion is needed here.

    Returns:
        The path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name()}.json")
    with open(path, "w") as f:
        json.dump(
            {"hyperparameters": hyperparameters, "results": results},
            f,
            indent=2,
        )
    print(f"saved eval results to {path}")
    return path
