"""Compare ADMM against flat MPPI/PS baselines on the identical task.

Reports the OI-MPPI paper's own metrics (`Documentation/IROS2026.pdf`,
Sec. VI, via `oim.utils.metrics`): success rate, position error (all
trials and successful trials only), mean planning frequency, execution
time. Every method runs the same `PushT(clutter=True, robot="point")`
task with the same robot-level sampler budget, so ADMM's only advantage
over the flat baselines is the object-level reference and consensus.

Scope: trials vary only the RNG seed, not starting pose (`PushT(clutter=
True)` has one fixed start today); `robot="point"` only.

    uv run python examples/ablation_pusht.py --trials 5 --steps 200
"""

import argparse
import json
import os
from copy import deepcopy
from datetime import datetime
from typing import Any, Dict

import mujoco

from oim import ROOT
from oim.algs import (
    ADMM,
    MPPI,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.sim3d.run import run_3d_admm, run_3d_plain
from oim.tasks.pusht import PushT
from oim.utils.metrics import aggregate_metrics

PLAN_DT = 0.05
HORIZON = 15
NUM_SAMPLES = 64
NOISE_LEVEL = 0.5
TEMPERATURE = 0.5
NUM_KNOTS = 4
PLAN_HORIZON = HORIZON * PLAN_DT


def _exec_model_and_data(task: PushT) -> Any:
    """The fine-timestep execution model/data, matching `examples/pusht.py`."""
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.002
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
    mj_data.qpos[:] = [0.0, 0.0, 0.0, -0.05, -0.06]
    return mj_model, mj_data


def _run_admm(task: PushT, seed: int, steps: int) -> Dict[str, Any]:
    robot_optimizer = MPPI(
        task,
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        spline_type="linear",
        num_knots=NUM_KNOTS,
        seed=seed,
    )
    object_optimizer = MPPI(
        make_object_shim(task, dt=PLAN_DT),
        num_samples=NUM_SAMPLES,
        noise_level=NOISE_LEVEL,
        temperature=TEMPERATURE,
        plan_horizon=PLAN_HORIZON,
        spline_type="zero",
        num_knots=HORIZON,
        seed=seed,
    )
    consensus = WrenchConsensus(
        max_dual=2.0 * float(task.consensus_scale()[0]),
        scale=task.consensus_scale(),
    )
    ctrl = ADMM(
        task,
        robot_optimizer,
        object_optimizer,
        consensus,
        n_admm=8,
        eps_r=0.5,
        eps_s=0.5,
        proximal_weight=0.1,
        rho_init=10.0,
        noise_min=0.0,
        noise_kappa=0.0,
        noise_max=0.0,
        debug_print=False,
    )
    mj_model, mj_data = _exec_model_and_data(task)
    return run_3d_admm(
        task,
        ctrl,
        ctrl.init_params(seed=seed),
        mj_model,
        mj_data,
        frequency=1.0 / PLAN_DT,
        max_steps=steps,
        verbose=False,
    )


def _run_flat(
    method: str, task: PushT, seed: int, steps: int
) -> Dict[str, Any]:
    common = dict(
        num_samples=NUM_SAMPLES,
        plan_horizon=PLAN_HORIZON,
        spline_type="linear",
        num_knots=NUM_KNOTS,
        seed=seed,
    )
    if method == "mppi":
        ctrl = MPPI(
            task, noise_level=NOISE_LEVEL, temperature=TEMPERATURE, **common
        )
    elif method == "ps":
        ctrl = PredictiveSampling(task, noise_level=NOISE_LEVEL, **common)
    else:
        raise ValueError(f"unknown baseline '{method}'")
    mj_model, mj_data = _exec_model_and_data(task)
    return run_3d_plain(
        task,
        ctrl,
        ctrl.init_params(seed=seed),
        mj_model,
        mj_data,
        frequency=1.0 / PLAN_DT,
        max_steps=steps,
        verbose=False,
    )


def main() -> None:
    """Run `--trials` seeds of each method and print/save the comparison."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--trials", type=int, default=5)
    parser.add_argument("--steps", type=int, default=200)
    parser.add_argument("--seed0", type=int, default=0)
    args = parser.parse_args()

    task = PushT(
        clutter=True,
        planning_dt=PLAN_DT,
        robot="point",
        consensus_source="contact",
    )
    runners = {
        "admm": lambda seed: _run_admm(task, seed, args.steps),
        "mppi": lambda seed: _run_flat("mppi", task, seed, args.steps),
        "ps": lambda seed: _run_flat("ps", task, seed, args.steps),
    }

    summary: Dict[str, Any] = {}
    for method, run in runners.items():
        print(f"--- {method} ---")
        logs = []
        for i in range(args.trials):
            seed = args.seed0 + i
            log = run(seed)
            print(
                f"  trial {i} (seed={seed}): reached={log['reached']} "
                f"final pos_err={log['pos_err'][-1]:.4f}"
            )
            logs.append(log)
        metrics = aggregate_metrics(logs, dt=PLAN_DT)
        summary[method] = metrics
        print(f"  {metrics}")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(ROOT, "results", "results_eval")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, f"pusht3d_point_ablation_{timestamp}.json")
    with open(path, "w") as f:
        json.dump(
            {
                "hyperparameters": dict(
                    trials=args.trials,
                    steps=args.steps,
                    seed0=args.seed0,
                    num_samples=NUM_SAMPLES,
                    noise_level=NOISE_LEVEL,
                    temperature=TEMPERATURE,
                    num_knots=NUM_KNOTS,
                    plan_horizon=PLAN_HORIZON,
                ),
                "results": summary,
            },
            f,
            indent=2,
        )
    print(f"saved ablation to {path}")


if __name__ == "__main__":
    main()
