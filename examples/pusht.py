import argparse
import math
import os
from copy import deepcopy
from datetime import datetime

import mujoco
import numpy as np

from oim import ROOT
from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.objects import Box, Circle, Polygon, rotate, t_shape_footprint
from oim.sim3d.deterministic import run_interactive
from oim.sim3d.run import run_3d_admm
from oim.tasks.pusht import CLUTTER_OBSTACLES, GOAL, PushT
from oim.utils.results import save_run_results

"""
Run an interactive simulation of the push-T task.
"""


def _base_name(robot: str, method: str) -> str:
    """`pusht3d_{robot}_{method}_{timestamp}`, no extension.

    Same scheme `examples/pusht2d.py` uses
    (`pusht2d_{scenario}_{method}_{timestamp}`), computed once and reused
    for every output of one run (plot/results), so they pair up under the
    same timestamp instead of each getting its own.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pusht3d_{robot}_{method}_{timestamp}"


def _obstacle_outline(obs: object, n: int = 48) -> np.ndarray:
    """A closed polyline tracing an obstacle, for filling in matplotlib.

    Same shapes `examples/pusht2d.py` draws (`oim.objects`), since
    `CLUTTER_OBSTACLES` is the same geometry the 2D `clutter` scenario uses.
    """
    if isinstance(obs, Circle):
        ang = np.linspace(0, 2 * np.pi, n)
        return np.asarray(obs.center) + obs.radius * np.stack(
            [np.cos(ang), np.sin(ang)], axis=1
        )
    if isinstance(obs, Polygon):
        return np.asarray(obs.vertices)
    if isinstance(obs, Box):
        he = np.asarray(obs.half_extents)
        corners = (
            np.array([[-1, -1], [1, -1], [1, 1], [-1, 1]], dtype=float) * he
        )
        return np.asarray(obs.center) + np.asarray(
            rotate(obs.angle, corners)
        )
    raise TypeError(f"no outline for {type(obs).__name__}")


def _footprint_world(pose: np.ndarray) -> np.ndarray:
    """The T-block's footprint at a given SE(2) pose, in world coordinates."""
    verts = np.asarray(t_shape_footprint().vertices)
    return np.asarray(pose[:2]) + np.asarray(rotate(float(pose[2]), verts))


def plot_run(log: dict, path: str, stride: int = 5) -> None:
    """Draw the block's swept footprint, the pusher path, and diagnostics.

    The 3D counterpart of `examples/pusht2d.py::plot_run` -- same layout,
    fed from `run_3d_admm`'s log instead of `run_2d`'s.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    for obs in CLUTTER_OBSTACLES.shapes:
        poly = _obstacle_outline(obs)
        ax.fill(poly[:, 0], poly[:, 1], color="0.4", zorder=1)
    goal = _footprint_world(np.asarray(GOAL))
    closed = np.vstack([goal, goal[:1]])
    ax.plot(
        closed[:, 0], closed[:, 1], color="green", lw=2, label="goal", zorder=4
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    poses = log["block_pose"]
    n = len(poses)
    for i in range(0, n, stride):
        w = _footprint_world(poses[i])
        ax.fill(
            w[:, 0],
            w[:, 1],
            color=plt.cm.viridis(i / max(n - 1, 1)),
            alpha=0.35,
            zorder=2,
        )
    for pose, colour in ((poses[0], "tab:blue"), (poses[-1], "tab:red")):
        w = _footprint_world(pose)
        ax.fill(w[:, 0], w[:, 1], color=colour, alpha=0.9, zorder=3)

    pusher = log["pusher_pos"]
    ax.plot(
        pusher[:, 0], pusher[:, 1], "k.-", ms=3, lw=1, label="pusher", zorder=5
    )
    ax.set_title(
        f"{'reached' if log['reached'] else 'not reached'} in {n - 1} steps"
    )
    ax.legend(loc="upper left")

    ax_r.plot(log["primal_residual"], label="primal residual")
    ax_r.plot(log["dual_residual"], label="dual residual")
    ax_r.plot(log["rho"], label="rho")
    ax_r.plot(np.linalg.norm(log["wrench"], axis=1), label="|w_rob| (N)")
    ax_r.set_xlabel("control step")
    ax_r.legend()
    ax_r.grid(alpha=0.3)
    ax_r.set_title("ADMM diagnostics")

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


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
admm_parser.add_argument(
    "--headless",
    action="store_true",
    help="Run without the interactive viewer, for a fixed number of steps "
    "(--steps), and save a trajectory/residual diagnostics plot instead "
    "of (optionally) recording a video. Needed for automated/no-display "
    "runs, since --record still requires a live viewer otherwise.",
)
admm_parser.add_argument(
    "--steps",
    type=int,
    default=200,
    help="Real control steps, --headless only.",
)
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
        # Residual-driven noise annealing (Algorithm 4 step 8) is off: the
        # primal residual never converges for this task (measured, over a
        # real 600-step run), so noise = clip(kappa*residual, min, max)
        # just sits pinned near noise_max the whole time, adding
        # near-constant large perturbations instead of annealing anything
        # -- confirmed directly to make divergence worse (final pos_err
        # 4.65 with it on vs. 2.0 with it off, same seed/config otherwise).
        noise_min=0.0,
        noise_kappa=0.0,
        noise_max=0.0,
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

    if args.headless:
        log = run_3d_admm(
            task, ctrl, ctrl.init_params(seed=args.seed), mj_model, mj_data,
            frequency=1.0 / plan_dt, max_steps=args.steps,
        )
        base = _base_name(args.robot, "admm")
        save_run_results(
            os.path.join(ROOT, "results"),
            base,
            hyperparameters=dict(
                robot=args.robot,
                steps=args.steps,
                n_admm=args.n_admm,
                robot_opt=args.robot_opt,
                object_opt=args.object_opt,
                rho=args.rho,
                gamma=args.gamma,
                seed=args.seed,
            ),
            log=log,
        )
        out_dir = os.path.join(ROOT, "recordings")
        os.makedirs(out_dir, exist_ok=True)
        plot_run(log, os.path.join(out_dir, f"{base}.png"))
    else:
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=1.0 / plan_dt,
            show_traces=False,
            record_video=args.record,
            recording_prefix=f"pusht3d_{args.robot}_admm",
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
        recording_prefix=f"pusht3d_{args.robot}_{args.algorithm or 'ps'}",
    )
