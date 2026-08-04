"""Config-driven launcher for push-T runs.

Same coverage as `examples/pusht.py`, but every hyperparameter comes from
`--robot`'s YAML config (`oim/configs/{robot}.yaml`), not a CLI
default -- CLI flags still override it for one-off changes. One config per
robot (not per environment): `--env` picks which 3D scene variant to run
(`clutter`, `gym2`, ...) against the *same* hyperparameters, so comparing
two environments means comparing the same sampler budget and tuning, not
whatever a separate per-environment config happened to set. Own code, not a
caller of `examples/pusht.py`; `--world 2d` only ever runs `admm`.

    uv run python -m oim.run_launch admm
    uv run python -m oim.run_launch --world 3d --robot xarm6 mppi --headless
    uv run python -m oim.run_launch admm --eval --trials 5
    uv run python -m oim.run_launch --world 2d admm
    uv run python -m oim.run_launch --robot xarm6 --env gym2 admm --headless
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
import yaml

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
from oim.objects import Box, Capsule, Circle, Polygon, rotate
from oim.sim2d import PushT2D, Scenario, build_scenario, run_2d
from oim.sim3d.deterministic import run_interactive
from oim.sim3d.run import run_3d_admm, run_3d_plain
from oim.tasks.pusht import PushT
from oim.utils.metrics import aggregate_metrics
from oim.utils.results import RunName, save_run_metrics, save_run_states
from oim.utils.results_eval import save_eval_results

# See examples/pusht.py's identical workaround: XLA's GPU command buffers
# leak across long closed loops and hit RESOURCE_EXHAUSTED.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
)

SUB_OPTIMIZERS = ["mppi", "cem", "ps", "cbo"]
CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")

# See examples/pusht.py: found via
# oim/models/xarm6_pusht_clutter/verify_reach.py.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]


def load_config(env: str) -> Dict[str, Any]:
    """Load `oim/configs/{env}.yaml`."""
    path = os.path.join(CONFIG_DIR, f"{env}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# Plotting -- same shapes as examples/pusht.py's, duplicated not imported.


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
    clutter-scene constant), so this plots correctly for any 3D scene
    variant (`clutter`, `gym2`, ...).
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
    """Write an animated gif of a 2D run."""
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


# Building tasks/controllers -- every parameter from `cfg`, not a literal.


def build_sub_optimizer(
    name: str,
    task: object,
    cfg: Dict[str, Any],
    *,
    plan_horizon: float,
    num_knots: int,
    spline: str,
    seed: int,
    num_samples: int,
) -> SamplingBasedController:
    """Build one sub-optimizer by name, its own params from `cfg["sampler"]`.

    `num_samples` is a parameter here (not read from `cfg` internally) so
    it flows through `args.samples` exactly like `plan_horizon` already
    does, with no separate cfg-mutation step needed for a CLI override.
    """
    common = dict(
        plan_horizon=plan_horizon, spline_type=spline, num_knots=num_knots,
        seed=seed,
    )
    p = cfg["sampler"].get(name, {})
    if name == "mppi":
        return MPPI(
            task, num_samples=num_samples, noise_level=p["noise_level"],
            temperature=p["temperature"], **common,
        )
    if name == "cem":
        return CEM(
            task, num_samples=num_samples, num_elites=p["num_elites"],
            sigma_start=p["sigma_start"], sigma_min=p["sigma_min"], **common,
        )
    if name == "ps":
        return PredictiveSampling(
            task, num_samples=num_samples, noise_level=p["noise_level"],
            **common,
        )
    if name == "cbo":
        return CBO(
            task, num_samples=num_samples,
            initial_noise_level=p["initial_noise_level"],
            temperature=p["temperature"],
            consensus_weight=p["consensus_weight"],
            noise_weight=p["noise_weight"], step_size=p["step_size"],
            **common,
        )
    raise ValueError(f"unknown sub-optimizer '{name}'")


def _build_3d(
    method: str, args: argparse.Namespace, cfg: Dict[str, Any]
) -> tuple:
    """Task + controller + execution model/data for one 3D method.

    Same execution-model settings and sampler construction path regardless
    of `method` or single-run vs `--eval` -- `examples/pusht.py`'s flat
    single-run path used different numbers (timestep, noise level, plan
    horizon) than its own `--eval` path; collapsing both into one function
    reading from `cfg` is what makes every method comparable by
    construction rather than by remembering to keep two paths in sync.
    """
    robot = args.robot
    dt = cfg["world3d"]["planning_dt"]
    horizon = args.horizon
    knots = cfg["sampler"]["robot_num_knots"]
    spline = cfg["sampler"]["robot_spline"]
    # Which 3D scene variant -- "clutter" (default) or "gym2" (see
    # oim/tasks/pusht.py, only meaningful for robot="xarm6"). From args, not
    # cfg directly, so `--env gym2` on the command line actually takes
    # effect -- the hyperparameters (sampler/admm/run) stay exactly the same
    # regardless of which scene is selected; only the task/scene changes.
    env = args.env

    task = PushT(
        impl="warp" if args.warp else "jax", clutter=True, planning_dt=dt,
        robot=robot, env=env,
        # "contact" (point-mass only) reads the real constraint force;
        # "twist" infers wrench from motion and converges worse (task 10).
        consensus_source="contact" if robot == "point" else "twist",
    )

    if method == "admm":
        consensus = WrenchConsensus(
            max_dual=2.0 * float(task.consensus_scale()[0]),
            scale=task.consensus_scale(),
        )
        robot_opt = build_sub_optimizer(
            args.robot_opt, task, cfg, plan_horizon=horizon * dt,
            num_knots=knots, spline=spline, seed=args.seed,
            num_samples=args.samples,
        )
        object_opt = build_sub_optimizer(
            args.object_opt, make_object_shim(task, dt=dt), cfg,
            plan_horizon=horizon * dt, num_knots=horizon,
            spline=cfg["sampler"]["object_spline"], seed=args.seed,
            num_samples=args.samples,
        )
        ctrl = ADMM(
            task, robot_opt, object_opt, consensus, n_admm=args.n_admm,
            eps_r=cfg["admm"]["eps_r"], eps_s=cfg["admm"]["eps_s"],
            proximal_weight=args.gamma, rho_init=args.rho,
            noise_min=cfg["admm"]["noise_min"],
            noise_kappa=cfg["admm"]["noise_kappa"],
            noise_max=cfg["admm"]["noise_max"],
        )
    else:
        ctrl = build_sub_optimizer(
            method, task, cfg, plan_horizon=horizon * dt, num_knots=knots,
            spline=spline, seed=args.seed, num_samples=args.samples,
        )

    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = cfg["world3d"]["exec_timestep"]
    mj_model.opt.iterations = cfg["world3d"]["exec_iterations"]
    mj_model.opt.ls_iterations = cfg["world3d"]["exec_ls_iterations"]
    mj_data = mujoco.MjData(mj_model)
    if robot == "xarm6":
        mj_data.qpos[:5] = [math.radians(q) for q in XARM6_START_QPOS_DEG]
        mj_data.qpos[5:8] = [0.0, 0.0, 0.0]  # block
    else:
        mj_data.qpos[:] = [0.0, 0.0, 0.0, -0.05, -0.06]
    return task, ctrl, mj_model, mj_data


def _build_2d(cfg: Dict[str, Any]) -> tuple:
    """Task + scenario for the 2D world, physics from `cfg["world2d"]`."""
    w2 = cfg["world2d"]
    scenario = build_scenario(w2["env"])
    task = PushT2D(
        footprint=scenario.footprint, goal=scenario.goal,
        obstacles=None if w2["no_obstacles"] else scenario.obstacles,
        mass=w2["mass"], mu=w2["mu"], mu_c=w2["mu_c"], f_max=w2["f_max"],
        contact_actions=w2["contact_action"],
        relocate_contact=not w2["no_relocate"],
    )
    return task, scenario


def _admm_2d(
    task: PushT2D, args: argparse.Namespace, cfg: Dict[str, Any]
) -> ADMM:
    """2D ADMM controller.

    No named robot/object sub-optimizer choice yet --
    `oim.sim2d.run.build_admm_2d`'s own MPPI, tuned for 2D, is reused as-is.
    """
    from oim.sim2d.run import build_admm_2d  # noqa: PLC0415

    ctrl, _ = build_admm_2d(
        task, horizon=args.horizon, num_samples=args.samples,
        n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
        eps_r=cfg["admm"]["eps_r"], eps_s=cfg["admm"]["eps_s"],
        noise_min=cfg["admm"]["noise_min"],
        noise_kappa=cfg["admm"]["noise_kappa"],
        noise_max=cfg["admm"]["noise_max"], seed=args.seed,
    )
    return ctrl


def _run_3d_once(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """One recorded 3D run: interactive viewer, or --headless plot/JSON."""
    task, ctrl, mj_model, mj_data = _build_3d(args.algorithm, args, cfg)
    dt = cfg["world3d"]["planning_dt"]
    is_admm = args.algorithm == "admm"
    name_parts = ["pusht3d", args.env, args.robot, args.algorithm]
    if is_admm:
        name_parts += [args.robot_opt, args.object_opt]
    name = RunName(*name_parts)

    tol = dict(
        goal_pos_tol=cfg["run"]["goal_pos_tol"],
        goal_theta_tol=cfg["run"]["goal_theta_tol"],
    )
    if getattr(args, "headless", False):
        out_dir = os.path.join(ROOT, "recordings")
        results_dir = os.path.join(ROOT, "results")
        os.makedirs(out_dir, exist_ok=True)
        if is_admm:
            log = run_3d_admm(
                task, ctrl, ctrl.init_params(seed=args.seed), mj_model,
                mj_data, frequency=1.0 / dt, max_steps=args.steps,
                record_dir=out_dir if args.record else None,
                record_name=name(), show_plans=args.show_plans, **tol,
            )
        else:
            # run_3d_plain has no recording support at all -- --record is
            # silently a no-op for flat methods, same as examples/pusht.py.
            log = run_3d_plain(
                task, ctrl, ctrl.init_params(seed=args.seed), mj_model,
                mj_data, frequency=1.0 / dt, max_steps=args.steps, **tol,
            )
        hyperparameters = dict(
            env=args.env, robot=args.robot, algorithm=args.algorithm,
            steps=args.steps, samples=args.samples,
            horizon=args.horizon, seed=args.seed,
        )
        if is_admm:
            hyperparameters.update(
                robot_opt=args.robot_opt, object_opt=args.object_opt,
                n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
            )
        save_run_metrics(
            results_dir, name, hyperparameters=hyperparameters, log=log
        )
        save_run_states(
            results_dir, name, task, log,
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
            ctrl, mj_model, mj_data, frequency=1.0 / dt, show_traces=False,
            record_video=args.record, recording_prefix=name(),
            **(dict(show_plans=args.show_plans) if is_admm else {}),
        )


def _run_2d_once(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """One 2D run: closed loop, results JSON, plot (+ optional gif)."""
    task, scenario = _build_2d(cfg)
    ctrl = _admm_2d(task, args, cfg)
    print(f"scenario: {scenario.name} -- {scenario.description}")
    block_kind = (
        "contact action" if cfg["world2d"]["contact_action"]
        else "direct wrench"
    )
    print(
        f"object block: {block_kind}  (action dim "
        f"{task.object_action_dim}, consensus dim {task.consensus_dim})"
    )

    ctx = jax.disable_jit() if args.no_jit else nullcontext()
    with ctx:
        log = run_2d(
            task, ctrl, ctrl.init_params(seed=args.seed),
            object_pose0=scenario.object_pose0, robot_pos0=scenario.robot_pos0,
            max_steps=args.steps, jit=not args.no_jit,
            goal_pos_tol=cfg["run"]["goal_pos_tol"],
            goal_theta_tol=cfg["run"]["goal_theta_tol"],
        )

    op = log["object_pose"]
    goal_xy = np.asarray(scenario.goal[:2])
    err0 = float(np.linalg.norm(np.asarray(op[0])[:2] - goal_xy))
    errN = float(np.linalg.norm(np.asarray(op[-1])[:2] - goal_xy))
    pct = 100.0 * (1.0 - errN / err0) if err0 > 0 else 0.0
    print(f"position error {err0:.4f} -> {errN:.4f}  ({pct:.1f}% closer)")

    name = RunName("pusht2d", scenario.name, "admm")
    results_dir = os.path.join(ROOT, "results")
    save_run_metrics(
        results_dir, name,
        hyperparameters=dict(
            env=scenario.name, steps=args.steps, n_admm=args.n_admm,
            samples=args.samples, horizon=args.horizon,
            rho=args.rho, gamma=args.gamma, seed=args.seed,
            mass=cfg["world2d"]["mass"], mu=cfg["world2d"]["mu"],
            mu_c=cfg["world2d"]["mu_c"],
            contact_action=cfg["world2d"]["contact_action"],
            relocate_contact=not cfg["world2d"]["no_relocate"],
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


def _run_eval(args: argparse.Namespace, cfg: Dict[str, Any]) -> None:
    """Run `--trials` seeds per method and save aggregate stats.

    On 3D, every sub-optimizer in `SUB_OPTIMIZERS` runs as a flat baseline
    alongside ADMM (or alone, if `--algorithm` names one directly), all
    built by the same `_build_3d`, so the comparison is a fair one: same
    sampler budget, same execution model, differing only in whether ADMM's
    object-level hierarchy is in the loop.
    """
    dt = cfg["world3d"]["planning_dt"]
    runners: Dict[str, Any] = {}
    tol = dict(
        goal_pos_tol=cfg["run"]["goal_pos_tol"],
        goal_theta_tol=cfg["run"]["goal_theta_tol"],
    )

    if args.world == "3d":
        if args.algorithm == "admm":
            task, ctrl, mj_model, mj_data = _build_3d("admm", args, cfg)
            label = f"admm_{args.robot_opt}_{args.object_opt}"
            runners[label] = lambda seed: run_3d_admm(
                task, ctrl, ctrl.init_params(seed=seed), mj_model, mj_data,
                frequency=1.0 / dt, max_steps=args.steps, verbose=False,
                **tol,
            )
            flat_methods = SUB_OPTIMIZERS
        else:
            flat_methods = (args.algorithm,)
        for method in flat_methods:
            flat_task, flat_ctrl, flat_model, flat_data = _build_3d(
                method, args, cfg
            )
            runners[method] = _flat_3d_runner(
                flat_task, flat_ctrl, flat_model, flat_data, dt, args.steps,
                tol,
            )
        base_name = RunName("pusht3d", args.env, args.robot, "eval")
    else:
        task, scenario = _build_2d(cfg)
        ctrl = _admm_2d(task, args, cfg)
        runners["admm"] = lambda seed: run_2d(
            task, ctrl, ctrl.init_params(seed=seed),
            object_pose0=scenario.object_pose0, robot_pos0=scenario.robot_pos0,
            max_steps=args.steps, verbose=False, **tol,
        )
        base_name = RunName("pusht2d", scenario.name, "eval")

    summary: Dict[str, Any] = {}
    for label, run in runners.items():
        print(f"--- {label} ---")
        logs: List[dict] = []
        for i in range(args.trials):
            seed = args.seed0 + i
            log = run(seed)
            logs.append(log)
            print(
                f"  trial {i} (seed={seed}): reached={log['reached']} "
                f"final pos_err={log['pos_err'][-1]:.4f}"
            )
        metrics = aggregate_metrics(logs, dt=dt)
        summary[label] = metrics
        print(f"  {metrics}")

    hyperparameters = dict(
        env=args.env, world=args.world, trials=args.trials,
        steps=args.steps, seed0=args.seed0,
        samples=args.samples, horizon=args.horizon,
    )
    if args.algorithm == "admm":
        hyperparameters.update(
            n_admm=args.n_admm, rho=args.rho, gamma=args.gamma,
        )
    save_eval_results(
        os.path.join(ROOT, "results", "results_eval"),
        base_name, hyperparameters=hyperparameters, results=summary,
    )


def _flat_3d_runner(
    task: PushT,
    ctrl: SamplingBasedController,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    dt: float,
    steps: int,
    tol: Dict[str, float],
):
    """Bind a flat baseline's rollout inputs into a `seed -> log` closure."""

    def run(seed: int):
        return run_3d_plain(
            task, ctrl, ctrl.init_params(seed=seed), mj_model, mj_data,
            frequency=1.0 / dt, max_steps=steps, verbose=False, **tol,
        )

    return run


def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build the parser, defaults sourced from `cfg`.

    CLI flags exist to override it for one-off changes, not to supply the
    values in the first place.
    """
    w3, w2, sampler, admm, run = (
        cfg["world3d"], cfg["world2d"], cfg["sampler"], cfg["admm"], cfg["run"],
    )
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--world", choices=["2d", "3d"], default="3d")
    parser.add_argument("--warp", action="store_true")
    parser.add_argument("--record", action="store_true")
    parser.add_argument(
        "--robot", choices=["point", "xarm6"], default=w3["robot"],
        help="Also selects which oim/configs/{robot}.yaml loads.",
    )
    parser.add_argument(
        "--env", choices=["clutter", "gym2"], default="clutter",
        help="3D xarm6 only: which scene variant, same hyperparameters "
        "either way -- see oim/tasks/pusht.py's `env` parameter. Not read "
        "from cfg: a scene choice, not a hyperparameter.",
    )
    parser.add_argument("--samples", type=int, default=sampler["num_samples"])
    parser.add_argument("--horizon", type=int, default=sampler["horizon"])
    parser.add_argument(
        "--contact-action", action=argparse.BooleanOptionalAction,
        default=w2["contact_action"], help="2D only.",
    )
    parser.add_argument(
        "--relocate", action=argparse.BooleanOptionalAction,
        default=not w2["no_relocate"], help="2D only.",
    )
    parser.add_argument(
        "--obstacles", action=argparse.BooleanOptionalAction,
        default=not w2["no_obstacles"], help="2D only.",
    )
    parser.add_argument("--no-jit", action="store_true", help="2D only.")
    parser.add_argument("--animate", action="store_true", help="2D only.")
    parser.add_argument("--no-plot", action="store_true", help="2D only.")

    subparsers = parser.add_subparsers(dest="algorithm")
    for method in SUB_OPTIMIZERS:
        sp = subparsers.add_parser(method)
        sp.add_argument("--eval", action="store_true")
        sp.add_argument("--trials", type=int, default=run["trials"])
        sp.add_argument("--seed0", type=int, default=run["seed0"])
        sp.add_argument("--seed", type=int, default=run["seed"])
        sp.add_argument("--steps", type=int, default=run["steps"])
        # No --headless: the diagnostics plot/metrics need ADMM-only log
        # fields a flat baseline doesn't have -- use --eval instead.

    admm_parser = subparsers.add_parser("admm")
    admm_parser.add_argument(
        "--robot-opt", choices=SUB_OPTIMIZERS, default=admm["robot_opt"],
    )
    admm_parser.add_argument(
        "--object-opt", choices=SUB_OPTIMIZERS, default=admm["object_opt"],
    )
    admm_parser.add_argument("--n-admm", type=int, default=admm["n_admm"])
    admm_parser.add_argument("--rho", type=float, default=admm["rho"])
    admm_parser.add_argument("--gamma", type=float, default=admm["gamma"])
    admm_parser.add_argument("--seed", type=int, default=run["seed"])
    admm_parser.add_argument("--headless", action="store_true", help="3D only.")
    admm_parser.add_argument("--steps", type=int, default=run["steps"])
    admm_parser.add_argument(
        "--show-plans", action="store_true", help="3D only."
    )
    admm_parser.add_argument("--eval", action="store_true")
    admm_parser.add_argument("--trials", type=int, default=run["trials"])
    admm_parser.add_argument("--seed0", type=int, default=run["seed0"])
    return parser


def main() -> None:
    """Parse `--robot`, load its config, then parse the full CLI.

    One config per robot, not per environment/scene -- `--env` (a normal
    flag in `_build_parser`, not pre-parsed here) picks the scene, but reads
    its hyperparameters from whichever robot's config already loaded, so
    every scene for a given robot is compared under identical tuning.
    """
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--robot", choices=["point", "xarm6"], default="point")
    pre_args, _ = pre.parse_known_args()
    cfg = load_config(pre_args.robot)

    parser = _build_parser(cfg)
    args = parser.parse_args()

    if args.world == "2d" and args.algorithm != "admm":
        parser.error(
            "2D has no plain running_cost/terminal_cost -- 'admm' is the "
            "only algorithm valid with --world 2d."
        )
    if args.env == "gym2" and args.robot != "xarm6":
        parser.error("--env gym2 requires --robot xarm6.")
    if args.world == "2d" and args.algorithm == "admm":
        if args.robot_opt != cfg["admm"]["robot_opt"] or (
            args.object_opt != cfg["admm"]["object_opt"]
        ):
            parser.error(
                "--robot-opt/--object-opt aren't wired into 2D's ADMM "
                "construction yet -- leave both at the config default "
                "with --world 2d."
            )

    # cfg's 2D-only fields are overridden in place by any explicit CLI
    # flag, so _build_2d/_admm_2d only ever need to read cfg.
    cfg["world2d"]["contact_action"] = args.contact_action
    cfg["world2d"]["no_relocate"] = not args.relocate
    cfg["world2d"]["no_obstacles"] = not args.obstacles

    if getattr(args, "eval", False):
        _run_eval(args, cfg)
    elif args.world == "2d":
        _run_2d_once(args, cfg)
    else:
        _run_3d_once(args, cfg)


if __name__ == "__main__":
    main()
