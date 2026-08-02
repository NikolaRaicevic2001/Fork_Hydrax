"""Run push-T: 2D (analytic) or 3D (MuJoCo MJX), one recorded run or --eval.

    # 3D, ADMM, interactive viewer
    uv run python examples/pusht.py admm

    # 3D, xArm6, headless, save a plot + results JSON
    uv run python examples/pusht.py --world 3d --robot xarm6 admm --headless

    # 2D, ADMM, the harder scenarios
    uv run python examples/pusht.py --world 2d --env corridor admm

    # 5 trials of ADMM (+ flat MPPI/PS baselines on 3D), aggregate stats
    uv run python examples/pusht.py admm --eval --trials 5 --steps 200

2D has no plain `running_cost`/`terminal_cost` (`PushT2D` implements only
`ConsensusTask`), so `mppi`/`ps` are 3D only; `--world 2d` requires `admm`.
"""

import argparse
import math
import os
from contextlib import nullcontext
from copy import deepcopy
from typing import Any, Dict, List

import jax
import mujoco
import numpy as np

from oim import ROOT
from oim.alg_base import SamplingBasedController
from oim.algs import (
    ADMM,
    CBO,
    CEM,
    MPPI,
    PredictiveSampling,
    WrenchConsensus,
    make_object_shim,
)
from oim.objects import Box, Capsule, Circle, Polygon, rotate, t_shape_footprint
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
from oim.tasks.pusht import CLUTTER_OBSTACLES, GOAL, PushT
from oim.utils.metrics import aggregate_metrics
from oim.utils.results import RunName, save_run_metrics, save_run_states
from oim.utils.results_eval import save_eval_results

# XLA's GPU command buffers (CUDA graphs) leak across ~200 iterations of a
# closed loop and hit RESOURCE_EXHAUSTED; disabling them is XLA's own
# suggested fix. Scoped here, not `oim/__init__.py`: doing it globally
# breaks the Warp backend. Matters most for 2D's long closed loops, but is
# harmless for 3D too.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
)

SUB_OPTIMIZERS = ["mppi", "cem", "ps", "cbo"]

# A starting joint configuration (degrees) that puts the xArm6's stick tip
# near the block's initial position, found via
# oim/models/xarm6_pusht_clutter/verify_reach.py's reach sweep -- not the
# arm's own zero-config pose, which isn't anywhere near the block.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]


def build_sub_optimizer(
    name: str,
    task: object,
    *,
    plan_horizon: float,
    num_knots: int,
    spline: str,
    seed: int,
    num_samples: int = 64,
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
            task,
            num_samples=num_samples,
            noise_level=0.5,
            temperature=0.5,
            **common,
        )
    if name == "cem":
        return CEM(
            task,
            num_samples=num_samples,
            num_elites=8,
            sigma_start=0.5,
            sigma_min=0.1,
            **common,
        )
    if name == "ps":
        return PredictiveSampling(
            task, num_samples=num_samples, noise_level=0.5, **common
        )
    if name == "cbo":
        return CBO(
            task,
            num_samples=num_samples,
            initial_noise_level=0.5,
            temperature=0.5,
            consensus_weight=1.0,
            noise_weight=1.0,
            step_size=0.1,
            **common,
        )
    raise ValueError(f"unknown sub-optimizer '{name}'")


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


def plot_run_3d(log: dict, path: str, stride: int = 5) -> None:
    """Draw the block's swept footprint, the pusher path, and diagnostics."""
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    verts = np.asarray(t_shape_footprint().vertices)
    for obs in CLUTTER_OBSTACLES.shapes:
        poly = _obstacle_outline(obs)
        ax.fill(poly[:, 0], poly[:, 1], color="0.4", zorder=1)
    goal = _footprint_world(verts, np.asarray(GOAL))
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
    # Imported here, not at module scope: the backend must be selected
    # before pyplot is first imported, and keeping both local means a
    # --no-plot run never touches matplotlib at all.
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
    plan_dt = 0.05
    horizon = args.horizon

    # "contact" reads MuJoCo's real constraint force; "twist" infers the
    # wrench from object motion and measurably converges worse (task 10).
    # Only valid for the point mass.
    consensus_source = "contact" if args.robot == "point" else "twist"
    task = PushT(
        impl="warp" if args.warp else "jax",
        clutter=True,
        planning_dt=plan_dt,
        robot=args.robot,
        consensus_source=consensus_source,
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
        num_knots=4,
        spline="linear",
        seed=args.seed,
        num_samples=args.samples,
    )
    object_optimizer = build_sub_optimizer(
        args.object_opt,
        make_object_shim(task, dt=plan_dt),
        plan_horizon=horizon * plan_dt,
        num_knots=horizon,
        spline="zero",
        seed=args.seed,
        num_samples=args.samples,
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
        # primal residual never converges for this task, so
        # noise = clip(kappa*residual, min, max) just sits pinned near
        # noise_max the whole time -- confirmed to make divergence worse.
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
    return task, ctrl, mj_model, mj_data


def _run_3d_admm_once(args: argparse.Namespace) -> None:
    """One recorded ADMM run: interactive viewer, or --headless plot/JSON."""
    print(
        f"Running ADMM object-informed MPPI (cluttered scene): "
        f"robot={args.robot_opt}, object={args.object_opt}"
    )
    task, ctrl, mj_model, mj_data = _build_admm_3d(args)

    if args.headless:
        name = RunName(
            "pusht3d", args.robot, "admm", args.robot_opt, args.object_opt
        )
        out_dir = os.path.join(ROOT, "recordings")
        results_dir = os.path.join(ROOT, "results")
        os.makedirs(out_dir, exist_ok=True)
        log = run_3d_admm(
            task, ctrl, ctrl.init_params(seed=args.seed), mj_model, mj_data,
            frequency=1.0 / 0.05, max_steps=args.steps,
            record_dir=out_dir if args.record else None,
            record_name=name(),
            show_plans=args.show_plans,
        )
        save_run_metrics(
            results_dir,
            name,
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
        save_run_states(
            results_dir,
            name,
            task,
            log,
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
        plot_run_3d(log, os.path.join(out_dir, f"{name()}.png"))
    else:
        run_interactive(
            ctrl,
            mj_model,
            mj_data,
            frequency=1.0 / 0.05,
            show_traces=False,
            record_video=args.record,
            recording_prefix=f"pusht3d_{args.robot}_admm_{args.robot_opt}_{args.object_opt}",
            show_plans=args.show_plans,
        )


def _run_3d_flat_once(args: argparse.Namespace) -> None:
    """One interactive run of a flat (non-ADMM) baseline."""
    # planning_dt coarsens the planner's own timestep for xarm6 (10 substeps
    # per rollout instead of 50); only the planning model changes, execution
    # below still steps at 0.001s. xarm6 still needs clutter=True (no
    # non-cluttered xarm6 scene exists).
    planning_dt = 0.05 if args.robot == "xarm6" else None
    task = PushT(
        impl="warp" if args.warp else "jax",
        clutter=(args.robot == "xarm6"),
        robot=args.robot,
        planning_dt=planning_dt,
    )
    # xArm6's per-rollout collision cost is much higher than the point
    # mass's; the point mass's defaults exhaust an 11 GB GPU for the arm.
    num_samples = 16 if args.robot == "xarm6" else 128
    num_randomizations = 1 if args.robot == "xarm6" else 4

    if args.algorithm == "ps":
        print("Running predictive sampling")
        ctrl = PredictiveSampling(
            task, num_samples=num_samples, noise_level=0.4,
            num_randomizations=num_randomizations, plan_horizon=0.5,
            spline_type="zero", num_knots=6,
        )
    else:
        print("Running MPPI")
        ctrl = MPPI(
            task, num_samples=num_samples, noise_level=0.4, temperature=0.0005,
            num_randomizations=num_randomizations, plan_horizon=0.5,
            spline_type="zero", num_knots=6,
        )

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

    run_interactive(
        ctrl, mj_model, mj_data, frequency=50, show_traces=False,
        record_video=args.record,
        recording_prefix=f"pusht3d_{args.robot}_{args.algorithm}",
    )


def _build_flat_3d(method: str, args: argparse.Namespace, dt: float) -> tuple:
    """Flat baseline for `--eval`.

    Same cluttered task and execution model ADMM uses, so the comparison
    isolates the object-level hierarchy.
    """
    task = PushT(clutter=True, planning_dt=dt, robot=args.robot)
    ctrl = build_sub_optimizer(
        method, task, plan_horizon=args.horizon * dt, num_knots=4,
        spline="linear", seed=args.seed, num_samples=args.samples,
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
    return task, ctrl, mj_model, mj_data


def _flat_3d_runner(
    task: PushT,
    ctrl: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    dt: float,
    steps: int,
):
    """Bind a flat baseline's rollout inputs into a `seed -> log` closure."""

    def run(seed: int):
        return run_3d_plain(
            task, ctrl, ctrl.init_params(seed=seed), mj_model, mj_data,
            frequency=1.0 / dt, max_steps=steps, verbose=False,
        )

    return run


def _build_2d_task_and_scenario(args: argparse.Namespace) -> tuple:
    scenario = build_scenario(args.env)
    task = PushT2D(
        footprint=scenario.footprint,
        goal=scenario.goal,
        obstacles=None if args.no_obstacles else scenario.obstacles,
        contact_actions=args.contact_action,
        relocate_contact=not args.no_relocate,
    )
    return task, scenario


def _run_2d_admm_once(args: argparse.Namespace) -> None:
    """One 2D run: closed loop, results JSON, plot (+ optional gif)."""
    task, scenario = _build_2d_task_and_scenario(args)
    print(f"scenario: {scenario.name} -- {scenario.description}")
    ctrl, params = build_admm_2d(
        task, horizon=args.horizon, num_samples=args.samples,
        n_admm=args.n_admm, rho=args.rho, gamma=args.gamma, seed=args.seed,
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
            task, ctrl, params, object_pose0=scenario.object_pose0,
            robot_pos0=scenario.robot_pos0, max_steps=args.steps,
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
    results_dir = os.path.join(ROOT, "results")
    save_run_metrics(
        results_dir, name,
        hyperparameters=dict(
            env=scenario.name, steps=args.steps, n_admm=args.n_admm,
            samples=args.samples, horizon=args.horizon, rho=args.rho,
            gamma=args.gamma, seed=args.seed,
            contact_action=args.contact_action,
            relocate_contact=not args.no_relocate,
        ),
        log=log,
    )
    save_run_states(
        results_dir, name, task, log,
        extra_static=dict(
            scenario=scenario.name, robot="disc",
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


def _run_eval(args: argparse.Namespace) -> None:
    """Run `--trials` seeds per method and save aggregate stats.

    For `admm` on 3D, also runs flat MPPI/PS with the same sampler budget
    for comparison -- ADMM's only advantage over them is then the object-
    level reference and consensus, not a bigger search. `admm` on 2D and
    the flat subcommands alone report just their own stats (2D has no flat
    baseline to compare against; `PushT2D` implements no plain
    `running_cost`/`terminal_cost`).
    """
    dt = 0.05
    runners: Dict[str, Any] = {}

    if args.world == "3d":
        if args.algorithm == "admm":
            task, ctrl, mj_model, mj_data = _build_admm_3d(args)
            label = f"admm_{args.robot_opt}_{args.object_opt}"
            runners[label] = lambda seed: run_3d_admm(
                task, ctrl, ctrl.init_params(seed=seed), mj_model, mj_data,
                frequency=1.0 / dt, max_steps=args.steps, verbose=False,
            )
            flat_methods = ("mppi", "ps")
        else:
            flat_methods = (args.algorithm,)
        for method in flat_methods:
            flat_task, flat_ctrl, flat_model, flat_data = _build_flat_3d(
                method, args, dt
            )
            runners[method] = _flat_3d_runner(
                flat_task, flat_ctrl, flat_model, flat_data, dt, args.steps
            )
        base_name = RunName("pusht3d", args.robot, "eval")
    else:
        task, scenario = _build_2d_task_and_scenario(args)
        # No named sub-optimizer choice for 2D yet (see _run_2d_admm_once);
        # "admm" alone is unambiguous since only one combo exists.
        ctrl, _ = build_admm_2d(
            task, horizon=args.horizon, num_samples=args.samples,
            n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
        )
        runners["admm"] = lambda seed: run_2d(
            task, ctrl, ctrl.init_params(seed=seed),
            object_pose0=scenario.object_pose0, robot_pos0=scenario.robot_pos0,
            max_steps=args.steps, verbose=False,
        )
        base_name = RunName("pusht2d", scenario.name, "eval")

    summary: Dict[str, Any] = {}
    for label, run in runners.items():
        print(f"--- {label} ---")
        logs: List[dict] = []
        for i in range(args.trials):
            seed = args.seed0 + i
            log = run(seed)
            print(
                f"  trial {i} (seed={seed}): reached={log['reached']} "
                f"final pos_err={log['pos_err'][-1]:.4f}"
            )
            logs.append(log)
        metrics = aggregate_metrics(logs, dt=dt)
        summary[label] = metrics
        print(f"  {metrics}")

    save_eval_results(
        os.path.join(ROOT, "results", "results_eval"),
        base_name,
        hyperparameters=dict(
            world=args.world, trials=args.trials, steps=args.steps,
            seed0=args.seed0, samples=args.samples, horizon=args.horizon,
            n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
        ),
        results=summary,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=["2d", "3d"], default="3d")
    parser.add_argument(
        "--warp", action="store_true", help="3D only: MjWarp instead of JAX."
    )
    parser.add_argument(
        "--record", action="store_true",
        help="3D only: write an mp4 to oim/recordings/ (needs ffmpeg).",
    )
    parser.add_argument(
        "--robot", choices=["point", "xarm6"], default="point",
        help="3D only. xarm6 always implies the cluttered scene.",
    )
    parser.add_argument(
        "--env", choices=list_scenarios(), default=DEFAULT_SCENARIO,
        help=f"2D only: scenario (default: {DEFAULT_SCENARIO}).",
    )
    parser.add_argument(
        "--contact-action", action="store_true",
        help="2D only: object block decides [p, f_n, f_t], not the wrench.",
    )
    parser.add_argument(
        "--no-relocate", action="store_true",
        help="2D only: disable the global contact-point search.",
    )
    parser.add_argument(
        "--no-obstacles", action="store_true",
        help="2D only: strip the obstacles.",
    )
    parser.add_argument(
        "--no-jit", action="store_true",
        help="2D only: run eagerly, steppable in a debugger.",
    )
    parser.add_argument(
        "--animate", action="store_true", help="2D only: also write a gif."
    )
    parser.add_argument(
        "--no-plot", action="store_true", help="2D only: skip the plot."
    )
    parser.add_argument(
        "--samples", type=int, default=64, help="Rollouts per sub-optimizer."
    )
    parser.add_argument(
        "--horizon", type=int, default=15, help="Consensus horizon H."
    )

    subparsers = parser.add_subparsers(dest="algorithm")
    flat_algorithms = (
        ("ps", "Predictive Sampling (3D only)"),
        ("mppi", "MPPI (3D only)"),
    )
    for algo_name, algo_help in flat_algorithms:
        sp = subparsers.add_parser(algo_name, help=algo_help)
        sp.add_argument("--eval", action="store_true")
        sp.add_argument("--trials", type=int, default=5)
        sp.add_argument("--seed0", type=int, default=0)
        sp.add_argument("--seed", type=int, default=5)
        sp.add_argument("--n-admm", type=int, default=8)
        sp.add_argument("--rho", type=float, default=10.0)
        sp.add_argument("--gamma", type=float, default=0.1)
        sp.add_argument("--steps", type=int, default=200)

    admm_parser = subparsers.add_parser(
        "admm", help="ADMM-coordinated object-informed MPPI (2D or 3D)"
    )
    admm_parser.add_argument(
        "--robot-opt", choices=SUB_OPTIMIZERS, default="mppi",
        help="Sampling optimizer for the robot-level ADMM block.",
    )
    admm_parser.add_argument(
        "--object-opt", choices=SUB_OPTIMIZERS, default="mppi",
        help="Sampling optimizer for the object-level ADMM block.",
    )
    admm_parser.add_argument("--n-admm", type=int, default=8)
    admm_parser.add_argument("--rho", type=float, default=10.0)
    admm_parser.add_argument("--gamma", type=float, default=0.1)
    admm_parser.add_argument("--seed", type=int, default=5)
    admm_parser.add_argument(
        "--headless", action="store_true",
        help="3D only: no viewer, run --steps steps, save plot + results JSON.",
    )
    admm_parser.add_argument("--steps", type=int, default=200)
    admm_parser.add_argument(
        "--show-plans", action="store_true",
        help="3D only: overlay both ADMM blocks' predicted trajectories.",
    )
    admm_parser.add_argument(
        "--eval", action="store_true",
        help="Run --trials seeds and save aggregate stats (3D: + flat MPPI/PS)"
        " instead of one recorded run.",
    )
    admm_parser.add_argument("--trials", type=int, default=5)
    admm_parser.add_argument("--seed0", type=int, default=0)
    return parser


def main() -> None:
    """Parse CLI arguments and dispatch to the selected world/mode."""
    parser = _build_parser()
    args = parser.parse_args()

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

    if getattr(args, "eval", False):
        _run_eval(args)
    elif args.world == "2d":
        _run_2d_admm_once(args)
    elif args.algorithm == "admm":
        _run_3d_admm_once(args)
    else:
        _run_3d_flat_once(args)


if __name__ == "__main__":
    main()
