r"""Sweep driver: run `examples/pusht.py` once per parameter combination.

<<<<<<< HEAD
A benchmark or ablation is a cartesian product -- tasks x algorithms x
horizons x sampler budgets x seeds -- and this runs it. It holds no
planning code of its own: each cell is a subprocess invocation of
`examples/pusht.py`, so a cell is exactly a command you could have typed,
and the launcher prints that command before running it.

    # the whole product in oim/configs/run_launch_config.yaml
    uv run python -m oim.run_launch

    # see what would run, without running it
    uv run python -m oim.run_launch --dry-run

    # narrow a sweep without editing the config
    uv run python -m oim.run_launch --only algorithm=admm --only horizon=15

Scoring is deliberately elsewhere: this writes run files, and
`oim/run_eval.py` turns them into tables. A sweep is expensive and a metric
is cheap, so they must not share a lifetime.

Subprocess isolation is the point of not importing the runners directly: a
crashed or OOM cell loses one combination instead of the sweep, and each
run starts with a clean JAX allocator. The cost is a recompile per cell,
which is unavoidable anyway whenever the horizon or sample count changes
the traced shapes.
=======
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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f
"""

import argparse
import importlib.util
import itertools
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

import jax
import yaml

from oim import ROOT
<<<<<<< HEAD
=======
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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f

CONFIG_DIR = os.path.join(os.path.dirname(__file__), "configs")
PUSHT = os.path.join(os.path.dirname(ROOT), "examples", "pusht.py")

# Sweep axes, in nesting order: earlier axes vary slowest, so every cell of
# one task finishes before the next task starts.
_AXES = (
    "task",
    "algorithm",
    "robot_opt",
    "object_opt",
    "horizon",
    "samples",
    "n_admm",
    "seed",
)

# gym2's own starting config, for its own base mount (GYM2_XARM6_BASE_POS/
# YAW in oim/tasks/pusht.py -- IsaacGym's literal (0.4, 0), yaw 0). Not
# IsaacGym's own literal init_joint_pose either: tried directly reusing it
# ([0, -45, -45, 0, 90] degrees, parsed from xarm6_stick.yaml's joint/vel
# pairs) and it puts the tip at z=0.36 pointing straight up -- the two
# xArm6 URDF-to-MJCF conversions don't share a joint-zero convention, so
# raw angles don't transfer. Found instead by the same grid search as
# XARM6_START_QPOS_DEG should have used (scoring for margin from every
# joint's own range limit, not just tip position/tilt): tip within 0.037m
# of the block start (0.7, -0.45), 10-degree tilt, every joint within
# 36-64% of its own range (none near a limit).
GYM2_XARM6_START_QPOS_DEG = [-59.0, 31.0, -74.0, 10.0, 50.0]


def _flag_spec() -> Tuple[Dict[str, bool], Dict[str, bool]]:
    """Ask `examples/pusht.py` which flags it takes, and where.

    Its parser splits on the algorithm name -- world and scene flags before
    it, solver and run flags after -- and mixes store_true switches with
    valued options. Rather than restate that here (a copy that silently
    rots the moment a flag moves), the classification is read off the real
    parser: a wrong guess would otherwise surface as every cell of a long
    sweep failing identically.

    Returns:
        `(top_level, per_algorithm)`, each mapping a dest name to whether
        the flag takes a value (False means a bare switch).
    """
    # Importing pusht.py builds module-level `jnp` arrays (`GOAL`,
    # `CLUTTER_OBSTACLES`), which initializes JAX's GPU backend and claims
    # ~75% of the device -- in *this* process, which then holds it for the
    # whole sweep and starves every cell of it. The launcher only reads a
    # parser, so pin it to CPU first.
    #
    # `jax.config`, not `JAX_PLATFORMS`: the environment variable is read
    # when `jax` is first imported, and `oim/__init__.py` has already done
    # that by the time this runs. Setting it here would silently do
    # nothing. Children are unaffected either way -- they are separate
    # processes with their own JAX.
    jax.config.update("jax_platforms", "cpu")
    spec = importlib.util.spec_from_file_location("_pusht_cli", PUSHT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    parser = module._build_parser()

    top: Dict[str, bool] = {}
    sub: Dict[str, bool] = {}
    for action in parser._actions:
        if isinstance(action, argparse._SubParsersAction):
            for subparser in action.choices.values():
                for a in subparser._actions:
                    if a.dest != "help":
                        sub[a.dest] = a.nargs != 0
        elif action.dest not in ("help",):
            top[action.dest] = action.nargs != 0
    return top, sub


def load_config(path: str) -> Dict[str, Any]:
    """Load a sweep config.

    Args:
        path: Path to the YAML, or a bare name resolved in `oim/configs/`.

    Returns:
        The parsed config, with `sweep` and `fixed` keys.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    if not os.path.isabs(path) and not os.path.exists(path):
        path = os.path.join(CONFIG_DIR, path)
    if not path.endswith((".yaml", ".yml")):
        path += ".yaml"
    with open(path) as f:
        return yaml.safe_load(f)


def expand(sweep: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Expand the sweep axes into one dict per combination.

    Cells that cannot exist are dropped rather than left to fail at
    runtime: 2D has no flat baselines, and a flat baseline has no ADMM
    blocks, so sweeping `robot_opt` would otherwise re-run the identical
    baseline once per pair.

    Args:
        sweep: The config's `sweep` block, `{axis: [values]}`.

<<<<<<< HEAD
    Returns:
        One dict per valid combination, in nesting order.
=======

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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f
    """
    axes = [a for a in _AXES if sweep.get(a)]
    combos = []
    seen = set()
    for values in itertools.product(*(sweep[a] for a in axes)):
        cell = dict(zip(axes, values, strict=True))
        task = cell.get("task", {})
        world = task.get("world", "3d")
        algorithm = cell.get("algorithm", "admm")

        if world == "2d" and algorithm != "admm":
            continue  # PushT2D implements only ConsensusTask
        if algorithm != "admm":
            # Flat baselines have no blocks; collapse the duplicates.
            cell = {
                k: v
                for k, v in cell.items()
                if k not in ("robot_opt", "object_opt", "n_admm")
            }
        key = json.dumps(cell, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        combos.append(cell)
    return combos


def build_command(
    cell: Dict[str, Any],
    fixed: Dict[str, Any],
    spec: Optional[Tuple[Dict[str, bool], Dict[str, bool]]] = None,
) -> List[str]:
    """Turn one sweep cell into an `examples/pusht.py` command line.

    Args:
        cell: One combination from `expand`.
        fixed: The config's `fixed` block, applied to every cell.
        spec: `_flag_spec()` output; read once and reused across a sweep.

    Returns:
        The argv list, algorithm name in its required position.

    Raises:
        ValueError: If a config key is not a flag `examples/pusht.py`
            accepts -- caught here rather than as an argparse error
            repeated once per cell.
    """
<<<<<<< HEAD
    top, sub = spec if spec is not None else _flag_spec()
    task = dict(cell.get("task", {}))
    algorithm = cell.get("algorithm", "admm")
    settings = {
        **fixed,
        **{k: v for k, v in cell.items() if k not in ("task", "algorithm")},
        **{k: v for k, v in task.items()},
    }

    pre: List[str] = []
    post: List[str] = []
    for key, value in settings.items():
        flag = "--" + key.replace("_", "-")
        if key in top:
            target, takes_value = pre, top[key]
        elif key in sub:
            target, takes_value = post, sub[key]
=======
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
        start_qpos_deg = (
            GYM2_XARM6_START_QPOS_DEG if env == "gym2" else XARM6_START_QPOS_DEG
        )
        mj_data.qpos[:5] = [math.radians(q) for q in start_qpos_deg]
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
            # gym2's own fixed "front" camera (see xarm6_pusht_gym2.xml) --
            # matches the Object-Informed-Manipulation (IsaacGym) repo's
            # own reference framing. clutter has no named camera, so this
            # stays None (the default free camera) there.
            camera = "front" if args.env == "gym2" else None
            log = run_3d_admm(
                task, ctrl, ctrl.init_params(seed=args.seed), mj_model,
                mj_data, frequency=1.0 / dt, max_steps=args.steps,
                record_dir=out_dir if args.record else None,
                record_name=name(), show_plans=args.show_plans, camera=camera,
                **tol,
            )
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f
        else:
            raise ValueError(
                f"'{key}' is not an examples/pusht.py flag; check the sweep "
                f"config. Known: {sorted(set(top) | set(sub))}"
            )
<<<<<<< HEAD
        if takes_value:
            target += [flag, str(value)]
        elif value:
            target.append(flag)

    return [sys.executable, PUSHT, *pre, algorithm, *post]
=======
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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f


def _gpu_memory() -> Optional[Tuple[int, int]]:
    """`(free, total)` MiB on GPU 0, or None if there is no `nvidia-smi`.

    Returns:
        The reading, or None on a machine without an NVIDIA GPU -- the
        sweep must still run there, just without the memory barrier.
    """
    try:
        out = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free,memory.total",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    free, total = out.stdout.strip().split("\n")[0].split(",")
    return int(free), int(total)


def _await_gpu(
    min_free_mib: int = 0, timeout: float = 120.0, poll: float = 0.5
) -> Optional[int]:
    """Wait for the previous cell's GPU memory to come back.

    A cell is its own process, so the driver reclaims on exit -- measured
    at well under a second, since the process is only reaped after CUDA
    teardown. This barrier exists for the cases where that reasoning does
    not hold: a cell that crashed rather than exited, a viewer or notebook
    someone left on the card, or a second sweep started by mistake. The
    xArm6 ADMM cell peaks near 12 GB of 16, so a few hundred stray MiB is
    the difference between a sweep and a column of Warp OOMs.

    Args:
        min_free_mib: Require at least this much free before returning. 0
            waits only for the reading to settle.
        timeout: Give up waiting after this long and run anyway -- an
            unmet requirement is reported by the cell that fails, which is
            more informative than the launcher refusing to start.
        poll: Seconds between readings.

    Returns:
        Free MiB at the moment of returning, or None without a GPU.
    """
    reading = _gpu_memory()
    if reading is None:
        return None

    deadline = time.time() + timeout
    free, _ = reading
    previous = -1
    while time.time() < deadline:
        # Settled (two equal readings) and enough headroom: go.
        if free == previous and free >= min_free_mib:
            return free
        previous = free
        time.sleep(poll)
        free = _gpu_memory()[0]

    print(
        f"  warning: only {free} MiB free after waiting {timeout:.0f}s "
        f"(wanted {min_free_mib} MiB); running anyway"
    )
    return free


def _label(cell: Dict[str, Any]) -> str:
    """A short human-readable name for one cell, for progress output."""
    task = cell.get("task", {})
    parts = [task.get("world", "3d")]
    parts += [str(task[k]) for k in ("robot", "env") if k in task]
    parts.append(str(cell.get("algorithm", "admm")))
    parts += [
        f"{k}={cell[k]}"
        for k in ("horizon", "samples", "n_admm", "seed")
        if k in cell
    ]
    return " ".join(parts)


def run_sweep(
    combos: Sequence[Dict[str, Any]],
    fixed: Dict[str, Any],
    dry_run: bool = False,
    keep_going: bool = True,
    min_free_mib: int = 0,
    gpu_timeout: float = 120.0,
) -> List[Dict[str, Any]]:
    """Run every combination, reporting progress and collecting outcomes.

    Args:
        combos: Cells from `expand`.
        fixed: The config's `fixed` block.
        dry_run: Print the commands without running them.
        keep_going: Continue after a cell fails. On by default -- a sweep
            is long and one bad combination should not discard the rest.
        min_free_mib: GPU memory required before a cell starts; see
            `_await_gpu`.
        gpu_timeout: How long to wait for it.

    Returns:
        One record per cell: its settings, command, exit status, duration,
        and the free GPU memory it started with.
    """
    records = []
    spec = _flag_spec()
    for i, cell in enumerate(combos, 1):
        cmd = build_command(cell, fixed, spec)
        print(f"\n[{i}/{len(combos)}] {_label(cell)}")
        print("  " + " ".join(cmd))
        if dry_run:
            records.append({"cell": cell, "command": cmd, "status": "skipped"})
            continue

        # Between cells, not only after: this also catches memory held by
        # something that was not part of this sweep.
        free = _await_gpu(min_free_mib, gpu_timeout)
        if free is not None:
            print(f"  {free} MiB free")

        t0 = time.time()
        result = subprocess.run(cmd, check=False)
        elapsed = time.time() - t0
        ok = result.returncode == 0
        print(f"  {'ok' if ok else 'FAILED'} in {elapsed:.1f}s")
        records.append(
            {
                "cell": cell,
                "command": cmd,
                "status": "ok" if ok else "failed",
                "returncode": result.returncode,
                "seconds": elapsed,
                "gpu_free_mib_at_start": free,
            }
        )
        if not ok and not keep_going:
            print("  stopping (--stop-on-error)")
            break
    return records


def _parse_only(only: Sequence[str]) -> Dict[str, str]:
    """Parse repeated `--only key=value` filters."""
    return dict(o.split("=", 1) for o in only)


<<<<<<< HEAD
def _parse_overrides(overrides: Sequence[str]) -> Dict[str, Any]:
    """Parse repeated `--set key=value` into typed values.
=======
def _build_parser(cfg: Dict[str, Any]) -> argparse.ArgumentParser:
    """Build the parser, defaults sourced from `cfg`.
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f

    Values go through the YAML loader, so `true`, `50` and `0.1` arrive as
    a bool, an int and a float rather than as strings -- `warp=false` has
    to be falsy, or switching a `fixed:` entry off from the command line
    would silently switch it on.

    Args:
        overrides: Raw `key=value` strings.

    Returns:
        The parsed overrides.

    Raises:
        ValueError: If an entry has no `=`.
    """
<<<<<<< HEAD
    parsed: Dict[str, Any] = {}
    for item in overrides:
        if "=" not in item:
            raise ValueError(f"--set expects KEY=VALUE, got '{item}'")
        key, value = item.split("=", 1)
        parsed[key.strip()] = yaml.safe_load(value)
    return parsed
=======
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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f


def _check_fixed(fixed: Dict[str, Any]) -> None:
    """Fail early if a `fixed:` key is not an `examples/pusht.py` flag.

    Otherwise a typo surfaces as every cell of a long sweep failing
    identically, minutes apart, with an argparse error buried in each
    subprocess's output.

    Args:
        fixed: The merged `fixed:` block and CLI overrides.

    Raises:
        ValueError: If any key is unknown.
    """
    top, sub = _flag_spec()
    unknown = sorted(set(fixed) - set(top) - set(sub))
    if unknown:
        raise ValueError(
            f"not examples/pusht.py flags: {unknown}. "
            f"Known: {sorted(set(top) | set(sub))}"
        )


def _apply_only(
    combos: List[Dict[str, Any]], only: Dict[str, str]
) -> List[Dict[str, Any]]:
    """Keep cells matching every filter, comparing as strings."""
    kept = []
    for cell in combos:
        flat = {**cell.get("task", {}), **cell}
        if all(str(flat.get(k)) == v for k, v in only.items()):
            kept.append(cell)
    return kept


def main() -> None:
<<<<<<< HEAD
    """Parse arguments, expand the sweep, run it, save a manifest."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--config",
        default="run_launch_config",
        help="Sweep config: a path, or a name under oim/configs/.",
    )
    p.add_argument(
        "--only",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keep only cells matching this; repeatable.",
    )
    p.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the commands and the cell count, run nothing.",
    )
    p.add_argument(
        "--stop-on-error",
        action="store_true",
        help="Abort the sweep on the first failing cell.",
    )
    p.add_argument(
        "--manifest-dir",
        default=os.path.join(ROOT, "results", "sweeps"),
        help="Where to write the record of what ran.",
    )
    p.add_argument(
        "--min-free-mib",
        type=int,
        default=0,
        help="Wait for at least this much free GPU memory before each cell. "
        "0 waits only for the reading to settle. The xArm6 ADMM cell peaks "
        "near 12000.",
    )
    p.add_argument(
        "--gpu-timeout",
        type=float,
        default=120.0,
        help="Seconds to wait for GPU memory before running anyway.",
    )
    p.add_argument(
        "--warp",
        action="store_true",
        help="Run every cell on the MuJoCo Warp backend "
        "(shorthand for --set warp=true).",
    )
    p.add_argument(
        "--set",
        action="append",
        default=[],
        dest="overrides",
        metavar="KEY=VALUE",
        help="Override a `fixed:` entry for this sweep only; repeatable. "
        "Any examples/pusht.py flag works, e.g. --set steps=50 "
        "--set record=false.",
    )
    args = p.parse_args()

    cfg = load_config(args.config)
    combos = expand(cfg.get("sweep", {}))
    if args.only:
        combos = _apply_only(combos, _parse_only(args.only))
=======
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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f

    fixed = dict(cfg.get("fixed", {}))
    if args.warp:
        fixed["warp"] = True
    try:
        fixed.update(_parse_overrides(args.overrides))
        _check_fixed(fixed)
    except ValueError as e:
        p.error(str(e))

    if not combos:
        print("no cells to run (check the sweep config and --only filters)")
        return

    print(f"{len(combos)} cells from {args.config}")
    t0 = time.time()
    records = run_sweep(
        combos,
        fixed,
        dry_run=args.dry_run,
        keep_going=not args.stop_on_error,
        min_free_mib=args.min_free_mib,
        gpu_timeout=args.gpu_timeout,
    )
    total = time.time() - t0

    failed = [r for r in records if r["status"] == "failed"]
    print(
        f"\n{len(records)} cells in {total / 60:.1f} min, {len(failed)} failed"
    )
    for r in failed:
        print(f"  FAILED: {_label(r['cell'])}")
    if args.dry_run:
        return

    os.makedirs(args.manifest_dir, exist_ok=True)
    stamp = time.strftime("%Y%m%d_%H%M%S")
    path = os.path.join(args.manifest_dir, f"sweep_{stamp}.json")
    with open(path, "w") as f:
        json.dump(
            {"config": args.config, "fixed": fixed, "runs": records},
            f,
            indent=2,
        )
<<<<<<< HEAD
    print(f"saved manifest to {path}")
    print("\nnext: uv run python -m oim.run_eval")
=======
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
>>>>>>> 6b734fa98acadff2ff75e7b9de00bfaa6ab74b9f


if __name__ == "__main__":
    main()
