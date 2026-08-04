"""Run push-T once: 2D (analytic) or 3D (MuJoCo MJX).

    # 3D, ADMM, interactive viewer
    uv run python examples/pusht.py admm

    # 3D, xArm6, headless, save a plot + results JSON
    uv run python examples/pusht.py --world 3d --robot xarm6 admm --headless

    # 2D, ADMM, the harder scenarios
    uv run python examples/pusht.py --world 2d --env corridor admm

One run per invocation. Sweeps are `oim/run_launch.py`'s job, and metrics
over the resulting run files are `oim/run_eval.py`'s -- neither is done
here, so a new metric never costs a re-run.

2D has no plain `running_cost`/`terminal_cost` (`PushT2D` implements only
`ConsensusTask`), so `mppi`/`ps` are 3D only; `--world 2d` requires `admm`.
"""

import argparse
import os
import sys
from contextlib import nullcontext
from copy import deepcopy
from typing import Optional

import jax
import mujoco
import numpy as np
import yaml

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
from oim.objects import Box, Capsule, Circle, Polygon, rotate
from oim.sim2d import (
    DEFAULT_SCENARIO,
    PushT2D,
    Scenario,
    build_admm_2d,
    build_scenario,
    list_scenarios,
    run_2d,
)
from oim.sim3d.deterministic import run_interactive
from oim.sim3d.run import run_3d_admm, run_3d_plain
from oim.tasks.pusht import PushT
from oim.utils.results import RunName, save_run
from oim.utils.scenes import SCENES

# XLA's GPU command buffers leak across long closed loops and hit
# RESOURCE_EXHAUSTED; disabling them is XLA's fix. Not in oim/__init__.py:
# doing it globally breaks the Warp backend.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
)

# Warp allocates outside JAX's pool, and JAX preallocates ~75% of the device
# the first time it touches it. On a 16 GB card that leaves Warp ~4 GB --
# not enough to build MuJoCo Warp's CUDA graphs for the xArm6 scene, which
# fails in `wp_cuda_graph_create_exec` before a single planning step. The
# two only coexist if JAX grows on demand instead.
#
# Read from argv rather than from parsed arguments: XLA fixes its allocator
# when the GPU backend first initializes, which happens long before main().
# `setdefault` so an explicit setting in the environment still wins.
if "--warp" in sys.argv:
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

SUB_OPTIMIZERS = ["mppi", "cem", "ps", "cbo"]

# Joint config (degrees) putting the xArm6's stick tip near the clutter
# scene's block start; found via
# oim/models/xarm6_pusht_clutter/verify_reach.py. Fallback only, and used
# by `clutter` alone -- a scene with its own workspace (the tabletop
# family) defines a "start" keyframe instead (see `_xarm6_start_qpos`),
# not another entry here.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]


def _xarm6_start_qpos(mj_model: mujoco.MjModel) -> np.ndarray:
    """Initial qpos for an xArm6 scene.

    Reads the model's own "start" keyframe if it defines one -- every
    scene besides the original clutter layout should, since each has a
    different workspace and needs its own tip-near-the-block starting
    pose -- falling back to `XARM6_START_QPOS_DEG` otherwise. Generic over
    `mj_model.nq` so it also covers the block's own qpos entries.
    """
    key_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_KEY, "start")
    if key_id >= 0:
        return np.array(mj_model.key_qpos[key_id])
    qpos = np.zeros(mj_model.nq)
    qpos[:5] = np.radians(XARM6_START_QPOS_DEG)
    return qpos


CONFIG_DIR = os.path.join(ROOT, "configs")


def load_config(robot: str, path: Optional[str] = None) -> dict:
    """Load the default parameters for a robot.

    Args:
        robot: `"point"` or `"xarm6"`, selecting `oim/configs/{robot}.yaml`.
        path: An explicit config file, overriding the robot lookup.

    Returns:
        The parsed config.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if path is None:
        path = os.path.join(CONFIG_DIR, f"{robot}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


def _named_camera(
    mj_model: mujoco.MjModel, name: str = "front"
) -> Optional[str]:
    """`name` if the model defines a camera by that name, else `None`.

    `None` means the default free camera framing the whole scene -- what
    every scene got before any of them defined their own. A scene with a
    layout the default camera frames badly (the tabletop family, viewed
    from the side by default) defines its own fixed camera instead of a
    special case here.
    """
    if mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, name) >= 0:
        return name
    return None


def build_sub_optimizer(
    name: str,
    task: object,
    *,
    plan_horizon: float,
    num_knots: int,
    spline: str,
    seed: int,
    num_samples: int,
    sampler_cfg: dict,
) -> object:
    """Build one sub-optimizer by name, from the config's block for it.

    Any `SamplingBasedController` works for either ADMM block -- the ADMM
    layer only ever calls `sample_knots`/`update_params` -- so the object-
    and robot-level optimizers are chosen independently here. Each one's
    own parameters come from `sampler_cfg[name]`, so retuning a method is
    a config edit rather than a code edit, and the same numbers are used
    whether it runs as an ADMM block or as a flat baseline.

    Args:
        name: `mppi`, `cem`, `ps` or `cbo`.
        task: The task to build against.
        plan_horizon: Planning horizon in seconds.
        num_knots: Spline knots.
        spline: Spline type.
        seed: RNG seed.
        num_samples: Rollouts per iteration.
        sampler_cfg: The config's `sampler` block.

    Returns:
        The controller.

    Raises:
        ValueError: If `name` is not a known sub-optimizer.
    """
    common = dict(
        plan_horizon=plan_horizon,
        spline_type=spline,
        num_knots=num_knots,
        seed=seed,
        num_samples=num_samples,
    )
    if name not in SUB_OPTIMIZERS:
        raise ValueError(f"unknown sub-optimizer '{name}'")
    own = sampler_cfg[name]
    if name == "mppi":
        return MPPI(task, **own, **common)
    if name == "cem":
        return CEM(task, **own, **common)
    if name == "ps":
        return PredictiveSampling(task, **own, **common)
    return CBO(task, **own, **common)


def _obstacle_outline(obs: object, n: int = 48) -> np.ndarray:
    """A closed polyline tracing an obstacle, for filling in matplotlib."""
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
        return np.asarray(obs.center) + np.asarray(rotate(obs.angle, corners))
    if isinstance(obs, Capsule):
        a, b = np.asarray(obs.a), np.asarray(obs.b)
        d = b - a
        ang = np.arctan2(d[1], d[0])
        t = np.linspace(-np.pi / 2, np.pi / 2, n // 2)
        cap_a = a + obs.radius * np.stack(
            [np.cos(t + ang + np.pi), np.sin(t + ang + np.pi)], axis=1
        )
        cap_b = b + obs.radius * np.stack(
            [np.cos(t + ang), np.sin(t + ang)], axis=1
        )
        return np.vstack([cap_a, cap_b])
    raise TypeError(f"no outline for {type(obs).__name__}")


def _footprint_world(verts: np.ndarray, pose: np.ndarray) -> np.ndarray:
    """The object's footprint at a given SE(2) pose, in world coordinates."""
    return np.asarray(pose[:2]) + np.asarray(rotate(float(pose[2]), verts))


def _diagnostics_panel(ax_r, log: dict) -> None:  # noqa: ANN001
    """Primal/dual residual, rho, and realized-wrench-norm traces."""
    ax_r.plot(log["primal_residual"], label="primal residual")
    ax_r.plot(log["dual_residual"], label="dual residual")
    ax_r.plot(log["rho"], label="rho")
    ax_r.plot(np.linalg.norm(log["wrench"], axis=1), label="|w_rob| (N)")
    ax_r.set_xlabel("control step")
    ax_r.legend()
    ax_r.grid(alpha=0.3)
    ax_r.set_title("ADMM diagnostics")


def plot_run_3d(task: PushT, log: dict, path: str, stride: int = 5) -> None:
    """Draw the block's swept footprint, the pusher path, and diagnostics.

    Obstacles/goal/footprint come from `task.object_model` (not a hardcoded
    clutter-scene constant), so this plots correctly for any `--scene`.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    verts = np.asarray(task.object_model.footprint.vertices)
    for obs in task.object_model.obstacles.shapes:
        poly = _obstacle_outline(obs)
        ax.fill(poly[:, 0], poly[:, 1], color="0.4", zorder=1)
    goal = _footprint_world(verts, np.asarray(task.object_model.goal))
    closed = np.vstack([goal, goal[:1]])
    ax.plot(
        closed[:, 0], closed[:, 1], color="green", lw=2, label="goal", zorder=4
    )
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    poses = log["object_pose"]
    n = len(poses)
    for i in range(0, n, stride):
        w = _footprint_world(verts, poses[i])
        ax.fill(
            w[:, 0],
            w[:, 1],
            color=plt.cm.viridis(i / max(n - 1, 1)),
            alpha=0.35,
            zorder=2,
        )
    for pose, colour in ((poses[0], "tab:blue"), (poses[-1], "tab:red")):
        w = _footprint_world(verts, pose)
        ax.fill(w[:, 0], w[:, 1], color=colour, alpha=0.9, zorder=3)

    pusher = log["robot_pos"]
    ax.plot(
        pusher[:, 0], pusher[:, 1], "k.-", ms=3, lw=1, label="pusher", zorder=5
    )
    ax.set_title(
        f"{'reached' if log['reached'] else 'not reached'} in {n - 1} steps"
    )
    ax.legend(loc="upper left")
    _diagnostics_panel(ax_r, log)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


def _draw_scene_2d(ax, scenario: Scenario, verts: np.ndarray) -> None:  # noqa: ANN001
    """Obstacles and the goal outline -- everything that does not move."""
    for obs in scenario.obstacles:
        poly = _obstacle_outline(obs)
        ax.fill(poly[:, 0], poly[:, 1], color="0.4", zorder=1)
    goal = _footprint_world(verts, np.asarray(scenario.goal))
    closed = np.vstack([goal, goal[:1]])
    ax.plot(
        closed[:, 0], closed[:, 1], color="green", lw=2, label="goal", zorder=4
    )
    ax.set_xlim(scenario.view[0], scenario.view[1])
    ax.set_ylim(scenario.view[2], scenario.view[3])
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)


def plot_run_2d(
    task: PushT2D, scenario: Scenario, log: dict, path: str, stride: int = 5
) -> None:
    """Draw the object's swept footprint, the robot path, and diagnostics."""
    # Local import: backend must be selected before pyplot loads, and
    # --no-plot then never touches matplotlib at all.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    verts = np.asarray(task.footprint.vertices)
    _draw_scene_2d(ax, scenario, verts)

    poses = log["object_pose"]
    n = len(poses)
    for i in range(0, n, stride):
        w = _footprint_world(verts, poses[i])
        ax.fill(
            w[:, 0],
            w[:, 1],
            color=plt.cm.viridis(i / max(n - 1, 1)),
            alpha=0.35,
            zorder=2,
        )
    for pose, colour in ((poses[0], "tab:blue"), (poses[-1], "tab:red")):
        w = _footprint_world(verts, pose)
        ax.fill(w[:, 0], w[:, 1], color=colour, alpha=0.9, zorder=3)

    robot = log["robot_pos"]
    ax.plot(
        robot[:, 0], robot[:, 1], "k.-", ms=3, lw=1, label="robot", zorder=5
    )
    ax.set_title(
        f"{scenario.name}  |  "
        f"{'reached' if log['reached'] else 'not reached'} in {n - 1} steps"
    )
    ax.legend(loc="upper left")
    _diagnostics_panel(ax_r, log)

    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"saved plot to {path}")


def save_animation_2d(
    task: PushT2D, scenario: Scenario, log: dict, path: str, fps: int = 15
) -> None:
    """Write an animated gif of a 2D run.

    Worth having over the static plot: a swept-footprint figure shows
    *where* the object went but not *when* it stalled, reversed, or got
    shoved sideways by a contact that broke.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import animation  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    verts = np.asarray(task.footprint.vertices)
    _draw_scene_2d(ax, scenario, verts)

    poses, robot = log["object_pose"], log["robot_pos"]
    (body,) = ax.fill([], [], color="tab:blue", alpha=0.85, zorder=3)
    (trail,) = ax.plot([], [], "k-", lw=1, alpha=0.6, zorder=5)
    (dot,) = ax.plot([], [], "ro", ms=5, zorder=6)
    title = ax.set_title("")

    def _update(i: int):  # noqa: ANN202
        body.set_xy(_footprint_world(verts, poses[i]))
        trail.set_data(robot[: i + 1, 0], robot[: i + 1, 1])
        dot.set_data([robot[i, 0]], [robot[i, 1]])
        title.set_text(f"{scenario.name}  step {i}/{len(poses) - 1}")
        return body, trail, dot, title

    anim = animation.FuncAnimation(
        fig, _update, frames=len(poses), blit=False, interval=1000 // fps
    )
    anim.save(path, writer=animation.PillowWriter(fps=fps))
    plt.close(fig)
    print(f"saved animation to {path}")


def _build_admm_3d(args: argparse.Namespace) -> tuple:
    """Task, ADMM controller, and execution model/data for the 3D world."""
    w3, smp, adm = args.cfg["world3d"], args.cfg["sampler"], args.cfg["admm"]
    plan_dt = w3["planning_dt"]
    horizon = args.horizon

    # "contact" (point-mass only) reads the real constraint force;
    # "twist" infers wrench from motion and converges worse (task 10).
    consensus_source = "contact" if args.robot == "point" else "twist"
    task = PushT(
        impl="warp" if args.warp else "jax",
        clutter=True,
        planning_dt=plan_dt,
        robot=args.robot,
        consensus_source=consensus_source,
        env=args.scene,
    )
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
        num_knots=smp["robot_num_knots"],
        spline=smp["robot_spline"],
        seed=args.seed,
        num_samples=args.samples,
        sampler_cfg=smp,
    )
    object_optimizer = build_sub_optimizer(
        args.object_opt,
        make_object_shim(task, dt=plan_dt),
        plan_horizon=horizon * plan_dt,
        num_knots=horizon,
        spline=smp["object_spline"],
        seed=args.seed,
        num_samples=args.samples,
        sampler_cfg=smp,
    )
    ctrl = ADMM(
        task,
        robot_optimizer,
        object_optimizer,
        consensus,
        n_admm=args.n_admm,
        eps_r=adm["eps_r"],
        eps_s=adm["eps_s"],
        proximal_weight=args.gamma,
        rho_init=args.rho,
        noise_min=adm["noise_min"],
        noise_kappa=adm["noise_kappa"],
        noise_max=adm["noise_max"],
    )

    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = w3["exec_timestep"]
    mj_model.opt.iterations = w3["exec_iterations"]
    mj_model.opt.ls_iterations = w3["exec_ls_iterations"]
    mj_data = mujoco.MjData(mj_model)
    if args.robot == "xarm6":
        mj_data.qpos[:] = _xarm6_start_qpos(mj_model)
    else:
        mj_data.qpos[:] = [0.0, 0.0, 0.0, -0.05, -0.06]
    return task, ctrl, mj_model, mj_data


def _run_3d_admm_once(args: argparse.Namespace) -> None:
    """One recorded ADMM run: interactive viewer, or --headless plot/JSON."""
    print(
        f"Running ADMM object-informed MPPI (cluttered scene): "
        f"robot={args.robot_opt}, object={args.object_opt}"
    )
    task, ctrl, mj_model, mj_data = _build_admm_3d(args)
    camera = _named_camera(mj_model)

    if args.headless:
        name = RunName(
            "pusht3d", args.robot, "admm", args.robot_opt, args.object_opt
        )
        out_dir = os.path.join(ROOT, "recordings")
        results_dir = os.path.join(ROOT, "results", "runs")
        os.makedirs(out_dir, exist_ok=True)
        log = run_3d_admm(
            task,
            ctrl,
            ctrl.init_params(seed=args.seed),
            mj_model,
            mj_data,
            frequency=1.0 / 0.05,
            max_steps=args.steps,
            goal_pos_tol=args.cfg["run"]["goal_pos_tol"],
            goal_theta_tol=args.cfg["run"]["goal_theta_tol"],
            record_dir=out_dir if args.record else None,
            record_name=name(),
            camera=camera,
            show_plans=args.show_plans,
        )
        save_run(
            results_dir,
            name,
            run=dict(
                world="3d",
                task=f"pusht3d_{args.robot}_{args.scene}",
                robot=args.robot,
                algorithm="admm",
                robot_opt=args.robot_opt,
                object_opt=args.object_opt,
                seed=args.seed,
            ),
            hyperparameters=dict(
                config=args.config_name,
                steps=args.steps,
                samples=args.samples,
                horizon=args.horizon,
                n_admm=args.n_admm,
                rho=args.rho,
                gamma=args.gamma,
                control_dt=0.05,
                goal_pos_tol=args.cfg["run"]["goal_pos_tol"],
                goal_theta_tol=args.cfg["run"]["goal_theta_tol"],
            ),
            task=task,
            log=log,
            extra_static=dict(
                robot=args.robot,
                sim_timestep=float(mj_model.opt.timestep),
                qpos_size=int(mj_model.nq),
                qvel_size=int(mj_model.nv),
                block_qpos_adr=(
                    task.block_qpos_adr if args.robot == "xarm6" else [0, 1, 2]
                ),
                block_dof_adr=task.block_dofs,
            ),
        )
        plot_run_3d(task, log, os.path.join(out_dir, f"{name()}.png"))
    else:
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=1.0 / 0.05,
            fixed_camera_id=(
                mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_CAMERA, camera)
                if camera
                else None
            ),
            show_traces=False,
            record_video=args.record,
            recording_prefix=f"pusht3d_{args.robot}_admm_{args.robot_opt}_{args.object_opt}",
            show_plans=args.show_plans,
        )


def _run_3d_flat_once(args: argparse.Namespace) -> None:
    """One run of a flat (non-ADMM) baseline: viewer, or --headless."""
    if args.headless:
        _run_3d_flat_headless(args)
        return
    # Coarsens the xarm6 planner's timestep only; execution still steps at
    # 0.001s. xarm6 always needs clutter=True (no non-cluttered scene).
    planning_dt = 0.05 if args.robot == "xarm6" else None
    task = PushT(
        impl="warp" if args.warp else "jax",
        clutter=(args.robot == "xarm6"),
        robot=args.robot,
        planning_dt=planning_dt,
        env=args.scene,
    )
    # xArm6's per-rollout collision cost is much higher than the point
    # mass's; the point mass's defaults exhaust an 11 GB GPU for the arm.
    num_samples = 16 if args.robot == "xarm6" else 128
    num_randomizations = 1 if args.robot == "xarm6" else 4

    if args.algorithm == "ps":
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
    else:
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

    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = 0.001
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
    if args.robot == "xarm6":
        mj_data.qpos[:] = _xarm6_start_qpos(mj_model)
    else:
        mj_data.qpos = [0.1, 0.1, 1.3, 0.0, 0.0]

    run_interactive(
        ctrl,
        mj_model,
        mj_data,
        frequency=50,
        show_traces=False,
        record_video=args.record,
        recording_prefix=f"pusht3d_{args.robot}_{args.algorithm}",
    )


def _run_3d_flat_headless(args: argparse.Namespace) -> None:
    """One recorded flat-baseline run, on the scene ADMM runs.

    Deliberately built by `_build_flat_3d`, not by the interactive path
    above: a baseline is only worth recording if it faces the same task,
    horizon, sampler budget and execution model ADMM does, and only this
    builder guarantees that.
    """
    dt = 0.05
    print(f"Running flat {args.algorithm} (cluttered scene, headless)")
    task, ctrl, mj_model, mj_data = _build_flat_3d(args.algorithm, args, dt)
    name = RunName("pusht3d", args.robot, args.algorithm)
    out_dir = os.path.join(ROOT, "recordings")
    os.makedirs(out_dir, exist_ok=True)

    log = run_3d_plain(
        task,
        ctrl,
        ctrl.init_params(seed=args.seed),
        mj_model,
        mj_data,
        frequency=1.0 / dt,
        max_steps=args.steps,
        goal_pos_tol=args.cfg["run"]["goal_pos_tol"],
        goal_theta_tol=args.cfg["run"]["goal_theta_tol"],
        # Same recording the ADMM path gets: a baseline you cannot watch
        # is a baseline you cannot debug.
        record_dir=out_dir if args.record else None,
        record_name=name(),
        camera=_named_camera(mj_model),
    )
    save_run(
        os.path.join(ROOT, "results", "runs"),
        name,
        run=dict(
            world="3d",
            task=f"pusht3d_{args.robot}_{args.scene}",
            robot=args.robot,
            algorithm=args.algorithm,
            robot_opt=args.algorithm,
            object_opt=None,
            seed=args.seed,
        ),
        hyperparameters=dict(
            config=args.config_name,
            steps=args.steps,
            samples=args.samples,
            horizon=args.horizon,
            n_admm=None,
            rho=None,
            gamma=None,
            control_dt=dt,
            goal_pos_tol=args.cfg["run"]["goal_pos_tol"],
            goal_theta_tol=args.cfg["run"]["goal_theta_tol"],
        ),
        task=task,
        log=log,
        extra_static=dict(
            robot=args.robot,
            sim_timestep=float(mj_model.opt.timestep),
            qpos_size=int(mj_model.nq),
            qvel_size=int(mj_model.nv),
            block_qpos_adr=(
                task.block_qpos_adr if args.robot == "xarm6" else [0, 1, 2]
            ),
            block_dof_adr=task.block_dofs,
        ),
    )


def _build_flat_3d(method: str, args: argparse.Namespace, dt: float) -> tuple:
    """Flat baseline on ADMM's own scene, horizon and sampler budget.

    Same cluttered task and execution model ADMM uses, so a comparison
    isolates the object-level hierarchy.
    """
    w3, smp = args.cfg["world3d"], args.cfg["sampler"]
    task = PushT(
        impl="warp" if args.warp else "jax",
        clutter=True,
        planning_dt=dt,
        robot=args.robot,
        env=args.scene,
    )
    ctrl = build_sub_optimizer(
        method,
        task,
        plan_horizon=args.horizon * dt,
        num_knots=smp["robot_num_knots"],
        spline=smp["robot_spline"],
        seed=args.seed,
        num_samples=args.samples,
        sampler_cfg=smp,
    )
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = w3["exec_timestep"]
    mj_model.opt.iterations = w3["exec_iterations"]
    mj_model.opt.ls_iterations = w3["exec_ls_iterations"]
    mj_data = mujoco.MjData(mj_model)
    if args.robot == "xarm6":
        mj_data.qpos[:] = _xarm6_start_qpos(mj_model)
    else:
        mj_data.qpos[:] = [0.0, 0.0, 0.0, -0.05, -0.06]
    return task, ctrl, mj_model, mj_data


def _build_2d_task_and_scenario(args: argparse.Namespace) -> tuple:
    w2 = args.cfg["world2d"]
    scenario = build_scenario(args.env)
    task = PushT2D(
        footprint=scenario.footprint,
        goal=scenario.goal,
        obstacles=None if args.no_obstacles else scenario.obstacles,
        contact_actions=args.contact_action,
        relocate_contact=not args.no_relocate,
        mass=w2["mass"],
        mu=w2["mu"],
        mu_c=w2["mu_c"],
        f_max=w2["f_max"],
    )
    return task, scenario


def _run_2d_admm_once(args: argparse.Namespace) -> None:
    """One 2D run: closed loop, results JSON, plot (+ optional gif)."""
    task, scenario = _build_2d_task_and_scenario(args)
    print(f"scenario: {scenario.name} -- {scenario.description}")
    ctrl, params = build_admm_2d(
        task,
        horizon=args.horizon,
        num_samples=args.samples,
        n_admm=args.n_admm,
        rho=args.rho,
        gamma=args.gamma,
        seed=args.seed,
    )
    block_kind = (
        "contact action [p, f_n, f_t]"
        if args.contact_action
        else "direct wrench"
    )
    print(
        f"object block: {block_kind}  (action dim {task.object_action_dim}, "
        f"consensus dim {task.consensus_dim})"
    )

    ctx = jax.disable_jit() if args.no_jit else nullcontext()
    with ctx:
        log = run_2d(
            task,
            ctrl,
            params,
            object_pose0=scenario.object_pose0,
            robot_pos0=scenario.robot_pos0,
            max_steps=args.steps,
            jit=not args.no_jit,
        )

    op = log["object_pose"]
    goal_xy = np.asarray(scenario.goal[:2])
    d0 = float(np.linalg.norm(op[0, :2] - goal_xy))
    d1 = float(np.linalg.norm(op[-1, :2] - goal_xy))
    pct = 100 * (1 - d1 / d0) if d0 > 0 else 0.0
    print(f"position error {d0:.4f} -> {d1:.4f}  ({pct:.1f}% closer)")

    # build_admm_2d has no named-sub-optimizer choice yet (unlike 3D's
    # --robot-opt/--object-opt, always MPPI with 2D-tuned noise levels),
    # so the name carries no sub-optimizer suffix -- there is only one.
    name = RunName("pusht2d", scenario.name, "admm")
    results_dir = os.path.join(ROOT, "results", "runs")
    save_run(
        results_dir,
        name,
        run=dict(
            world="2d",
            task=f"pusht2d_{scenario.name}",
            robot="disc",
            algorithm="admm",
            robot_opt="mppi",
            object_opt="mppi",
            seed=args.seed,
        ),
        hyperparameters=dict(
            config=args.config_name,
            steps=args.steps,
            samples=args.samples,
            horizon=args.horizon,
            n_admm=args.n_admm,
            rho=args.rho,
            gamma=args.gamma,
            control_dt=float(task.dt),
            goal_pos_tol=args.cfg["run"]["goal_pos_tol"],
            goal_theta_tol=args.cfg["run"]["goal_theta_tol"],
            contact_action=args.contact_action,
            relocate_contact=not args.no_relocate,
        ),
        task=task,
        log=log,
        extra_static=dict(
            scenario=scenario.name,
            robot="disc",
            robot_radius=float(task.model.robot_radius),
            robot_max_speed=float(task.u_max[0]),
        ),
    )

    if not args.no_plot:
        out_dir = os.path.join(ROOT, "recordings")
        os.makedirs(out_dir, exist_ok=True)
        plot_run_2d(task, scenario, log, os.path.join(out_dir, f"{name()}.png"))
        if args.animate:
            save_animation_2d(
                task, scenario, log, os.path.join(out_dir, f"{name()}.gif")
            )


def _build_parser(cfg: Optional[dict] = None) -> argparse.ArgumentParser:
    """Build the CLI, taking every default from `cfg`.

    Args:
        cfg: A config from `load_config`. Omitted (the launcher's
            introspection path) falls back to the point robot's file, since
            only the *shape* of the parser matters there.

    Returns:
        The parser.
    """
    if cfg is None:
        cfg = load_config("point")
    smp, adm, run = cfg["sampler"], cfg["admm"], cfg["run"]

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=["2d", "3d"], default="3d")
    parser.add_argument(
        "--config",
        default=None,
        help="Explicit parameter file; default oim/configs/{robot}.yaml.",
    )
    parser.add_argument(
        "--warp", action="store_true", help="3D only: MjWarp instead of JAX."
    )
    parser.add_argument(
        "--record",
        action="store_true",
        help="3D only: write an mp4 to oim/recordings/ (needs ffmpeg).",
    )
    parser.add_argument(
        "--robot",
        choices=["point", "xarm6"],
        default="point",
        help="3D only. xarm6 always implies the cluttered scene.",
    )
    parser.add_argument(
        "--env",
        choices=list_scenarios(),
        default=DEFAULT_SCENARIO,
        help=f"2D only: scenario (default: {DEFAULT_SCENARIO}).",
    )
    parser.add_argument(
        "--scene",
        choices=sorted(SCENES),
        default="clutter",
        help="3D only: which scene, from oim.utils.scenes.SCENES. Separate "
        "from --env, which names a 2D scenario. A scene not built for "
        "--robot errors at construction (PushT.__init__, via "
        "SceneSpec.mjcf_scene).",
    )
    parser.add_argument(
        "--contact-action",
        action="store_true",
        help="2D only: object block decides [p, f_n, f_t], not the wrench.",
    )
    parser.add_argument(
        "--no-relocate",
        action="store_true",
        help="2D only: disable the global contact-point search.",
    )
    parser.add_argument(
        "--no-obstacles",
        action="store_true",
        help="2D only: strip the obstacles.",
    )
    parser.add_argument(
        "--no-jit",
        action="store_true",
        help="2D only: run eagerly, steppable in a debugger.",
    )
    parser.add_argument(
        "--animate", action="store_true", help="2D only: also write a gif."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="2D only: skip the plot."
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=smp["num_samples"],
        help="Rollouts per sub-optimizer.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=smp["horizon"],
        help="Consensus horizon H.",
    )

    subparsers = parser.add_subparsers(dest="algorithm")
    flat_algorithms = (
        ("ps", "Predictive Sampling (3D only)"),
        ("mppi", "MPPI (3D only)"),
    )
    for algo_name, algo_help in flat_algorithms:
        sp = subparsers.add_parser(algo_name, help=algo_help)
        sp.add_argument("--seed", type=int, default=run["seed"])
        sp.add_argument(
            "--headless",
            action="store_true",
            help="No viewer: run --steps steps and save a run file.",
        )
        sp.add_argument("--n-admm", type=int, default=adm["n_admm"])
        sp.add_argument("--rho", type=float, default=adm["rho"])
        sp.add_argument("--gamma", type=float, default=adm["gamma"])
        sp.add_argument("--steps", type=int, default=run["steps"])

    admm_parser = subparsers.add_parser(
        "admm", help="ADMM-coordinated object-informed MPPI (2D or 3D)"
    )
    admm_parser.add_argument(
        "--robot-opt",
        choices=SUB_OPTIMIZERS,
        default=adm["robot_opt"],
        help="Sampling optimizer for the robot-level ADMM block.",
    )
    admm_parser.add_argument(
        "--object-opt",
        choices=SUB_OPTIMIZERS,
        default=adm["object_opt"],
        help="Sampling optimizer for the object-level ADMM block.",
    )
    admm_parser.add_argument("--n-admm", type=int, default=adm["n_admm"])
    admm_parser.add_argument("--rho", type=float, default=adm["rho"])
    admm_parser.add_argument("--gamma", type=float, default=adm["gamma"])
    admm_parser.add_argument("--seed", type=int, default=run["seed"])
    admm_parser.add_argument(
        "--headless",
        action="store_true",
        help="3D only: no viewer, run --steps steps, save plot + results JSON.",
    )
    admm_parser.add_argument("--steps", type=int, default=run["steps"])
    admm_parser.add_argument(
        "--show-plans",
        action="store_true",
        help="3D only: overlay both ADMM blocks' predicted trajectories.",
    )
    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the selected world/mode."""
    # Two stages, because which config supplies the defaults is itself a
    # CLI choice: peek at --robot/--config, load that file, then build the
    # real parser on top of it. Anything passed explicitly still wins --
    # argparse defaults are exactly "the value when the flag is absent".
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", default="point")
    pre.add_argument("--config", default=None)
    pre_args, _ = pre.parse_known_args()

    cfg = load_config(pre_args.robot, pre_args.config)
    parser = _build_parser(cfg)
    args = parser.parse_args()
    args.cfg = cfg
    # Provenance: which defaults produced this run, recorded alongside the
    # values themselves so a run file explains itself.
    args.config_name = pre_args.config or pre_args.robot

    if args.world == "2d" and args.algorithm != "admm":
        parser.error(
            "2D has no plain running_cost/terminal_cost -- 'admm' is the "
            "only algorithm valid with --world 2d (it is not implied; "
            "pass it explicitly, after --env and any other top-level "
            "flags)."
        )
    if args.world == "2d" and args.algorithm == "admm":
        if args.robot_opt != "mppi" or args.object_opt != "mppi":
            parser.error(
                "--robot-opt/--object-opt aren't wired into 2D's ADMM "
                "construction yet (build_admm_2d always uses MPPI, tuned "
                "with 2D-specific noise levels) -- leave both at the "
                "'mppi' default with --world 2d."
            )

    if args.algorithm is None:
        args.algorithm = "ps"

    scene_robots = SCENES[args.scene].mjcf_by_robot
    if args.world == "3d" and args.robot not in scene_robots:
        parser.error(
            f"--scene {args.scene} has no scene for --robot {args.robot} "
            f"(available: {sorted(scene_robots)})."
        )

    if args.world == "2d":
        _run_2d_admm_once(args)
    elif args.algorithm == "admm":
        _run_3d_admm_once(args)
    else:
        _run_3d_flat_once(args)


if __name__ == "__main__":
    main()
