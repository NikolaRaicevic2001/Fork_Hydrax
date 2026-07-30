import argparse
import math
from copy import deepcopy

import mujoco

from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.sim3d.deterministic import run_interactive
from oim.tasks.pusht import PushT

"""
Run an interactive simulation of the push-T task.
"""


def build_sub_optimizer(
    name: str,
    task: object,
    *,
    plan_horizon: float,
    num_knots: int,
    spline: str,
    seed: int,
) -> object:
    """Build one ADMM sub-optimizer by name.

    Any `SamplingBasedController` works for either ADMM block -- the ADMM
    layer only ever calls `sample_knots`/`update_params` -- so the object-
    and robot-level optimizers are chosen independently here.
    """
    common = dict(
        plan_horizon=plan_horizon,
        spline_type=spline,
        num_knots=num_knots,
        seed=seed,
    )
    if name == "mppi":
        return MPPI(
            task, num_samples=64, noise_level=0.5, temperature=0.5, **common
        )
    if name == "cem":
        return CEM(
            task,
            num_samples=64,
            num_elites=8,
            sigma_start=0.5,
            sigma_min=0.1,
            **common,
        )
    if name == "ps":
        return PredictiveSampling(
            task, num_samples=64, noise_level=0.5, **common
        )
    if name == "cbo":
        return CBO(
            task,
            num_samples=64,
            initial_noise_level=0.5,
            temperature=0.5,
            consensus_weight=1.0,
            noise_weight=1.0,
            step_size=0.1,
            **common,
        )
    raise ValueError(f"unknown sub-optimizer '{name}'")


SUB_OPTIMIZERS = ["mppi", "cem", "ps", "cbo"]

# A starting joint configuration (degrees) that puts the xArm6's stick tip
# near the block's initial position, found via
# oim/models/xarm6_pusht_clutter/verify_reach.py's reach sweep -- not
# the arm's own zero-config pose, which (after the base placement in
# oim/tasks/pusht.py's XARM6_BASE_POS/XARM6_BASE_YAW_DEG) isn't
# anywhere near the block.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]

# Parse command-line arguments
parser = argparse.ArgumentParser(
    description="Run an interactive simulation of the push-T task."
)
parser.add_argument(
    "--warp",
    action="store_true",
    help="Whether to use the (experimental) MjWarp backend. (default: False)",
)
parser.add_argument(
    "--record",
    action="store_true",
    help="Record an mp4 of the run to oim/recordings/ (needs ffmpeg).",
)
parser.add_argument(
    "--robot",
    choices=["point", "xarm6"],
    default="point",
    help=(
        "Which embodiment pushes the block: the original free 2-DOF point "
        "mass (default), or a real 6-DoF xArm6 (PushT's robot='xarm6'). "
        "xarm6 always implies the cluttered scene, even without the "
        "'admm' subcommand -- there is no non-cluttered xarm6 scene."
    ),
)
subparsers = parser.add_subparsers(
    dest="algorithm", help="Sampling algorithm (choose one)"
)
subparsers.add_parser("ps", help="Predictive Sampling")
subparsers.add_parser("mppi", help="Model Predictive Path Integral Control")
admm_parser = subparsers.add_parser(
    "admm", help="ADMM-coordinated object-informed MPPI on a cluttered scene"
)
admm_parser.add_argument(
    "--robot-opt",
    choices=SUB_OPTIMIZERS,
    default="mppi",
    help="Sampling optimizer for the robot-level ADMM block.",
)
admm_parser.add_argument(
    "--object-opt",
    choices=SUB_OPTIMIZERS,
    default="mppi",
    help="Sampling optimizer for the object-level ADMM block.",
)
admm_parser.add_argument("--n-admm", type=int, default=8)
admm_parser.add_argument("--rho", type=float, default=10.0)
admm_parser.add_argument("--gamma", type=float, default=0.1)
admm_parser.add_argument("--seed", type=int, default=5)
args = parser.parse_args()

impl = "warp" if args.warp else "jax"

if args.algorithm == "admm":
    plan_dt = 0.05
    horizon = 15  # consensus horizon H (steps of plan_dt)
    print(
        f"Running ADMM object-informed MPPI (cluttered scene): "
        f"robot={args.robot_opt}, object={args.object_opt}"
    )

    task = PushT(impl=impl, clutter=True, planning_dt=plan_dt, robot=args.robot)

    # Normalizing by the friction-cone limit keeps the ADMM penalty O(1)
    # and comparable to the task costs, so rho is a meaningful knob.
    consensus = WrenchConsensus(
        max_dual=2.0 * float(task.consensus_scale()[0]),
        scale=task.consensus_scale(),
    )

    robot_optimizer = build_sub_optimizer(
        args.robot_opt,
        task,
        plan_horizon=horizon * plan_dt,
        num_knots=4,
        spline="linear",
        seed=args.seed,
    )
    object_optimizer = build_sub_optimizer(
        args.object_opt,
        make_object_shim(task, dt=plan_dt),
        plan_horizon=horizon * plan_dt,
        num_knots=horizon,
        spline="zero",
        seed=args.seed,
    )
    ctrl = ADMM(
        task,
        robot_optimizer,
        object_optimizer,
        consensus,
        n_admm=args.n_admm,
        eps_r=0.5,
        eps_s=0.5,
        proximal_weight=args.gamma,
        rho_init=args.rho,
        noise_min=0.05,
        noise_kappa=0.1,
        noise_max=0.5,
    )

    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.002
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
    if args.robot == "xarm6":
        mj_data.qpos[:5] = [math.radians(q) for q in XARM6_START_QPOS_DEG]
        mj_data.qpos[5:8] = [0.0, 0.0, 0.0]  # block
    else:
        mj_data.qpos[:] = [0.0, 0.0, 0.0, -0.05, -0.06]

    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=1.0 / plan_dt,
        show_traces=False,
        record_video=args.record,
    )
else:
    # Plain (non-ADMM) MPC against the basic reach-and-touch running_cost.
    # xarm6 still needs clutter=True (no non-cluttered xarm6 scene exists).
    # planning_dt coarsens the planner's own timestep for xarm6 (10 substeps
    # per rollout instead of 50); only the planning model changes, execution
    # below still steps at 0.001s.
    planning_dt = 0.05 if args.robot == "xarm6" else None
    task = PushT(
        impl=impl,
        clutter=(args.robot == "xarm6"),
        robot=args.robot,
        planning_dt=planning_dt,
    )

    # xArm6's per-rollout collision cost is much higher than the point
    # mass's; the point mass's defaults (128 samples * 4 randomizations)
    # exhaust an 11 GB GPU for the arm.
    num_samples = 16 if args.robot == "xarm6" else 128
    num_randomizations = 1 if args.robot == "xarm6" else 4

    # Set the controller based on command-line arguments
    if args.algorithm == "ps" or args.algorithm is None:
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task,
            num_samples=num_samples,
            noise_level=0.4,
            num_randomizations=num_randomizations,
            plan_horizon=0.5,
            spline_type="zero",
            num_knots=6,
        )
    elif args.algorithm == "mppi":
        print("Running MPPI")
        ctrl = MPPI(
            task,
            num_samples=num_samples,
            noise_level=0.4,
            temperature=0.0005,
            num_randomizations=num_randomizations,
            plan_horizon=0.5,
            spline_type="zero",
            num_knots=6,
        )
    else:
        parser.error("Invalid algorithm")

    # Define the model used for simulation
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.001
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
    if args.robot == "xarm6":
        mj_data.qpos[:5] = [math.radians(q) for q in XARM6_START_QPOS_DEG]
        mj_data.qpos[5:8] = [0.0, 0.0, 0.0]  # block
    else:
        mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

    # Run the interactive simulation
    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
        record_video=args.record,
    )
