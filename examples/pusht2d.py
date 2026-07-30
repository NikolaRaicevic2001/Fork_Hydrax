"""Run a push task in the analytic 2D world.

The 2D counterpart of `examples/pusht.py admm`: same `ADMM` controller, same
consensus math, same object-level subproblem -- only the robot block's
physics is analytic instead of MuJoCo. Use it to isolate algorithm
behaviour from simulator behaviour, since anything that reproduces here is
the formulation rather than MJX.

    # T-block through clutter (the 2D twin of the MuJoCo demo)
    uv run python examples/pusht2d.py

    # push through a narrow channel / a gate slot
    uv run python examples/pusht2d.py --env corridor
    uv run python examples/pusht2d.py --env gate

    # A/B against the contact-action object parameterization
    uv run python examples/pusht2d.py --contact-action

    # step through the physics in a debugger (no jit anywhere)
    uv run python examples/pusht2d.py --no-jit --steps 3

    # write an animated gif as well as the summary plot
    uv run python examples/pusht2d.py --animate
"""

import argparse
import os
from contextlib import nullcontext
from datetime import datetime

import jax
import jax.numpy as jnp
import numpy as np

from oim import ROOT
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
from oim.utils.results import save_run_results

# XLA's GPU command buffers (CUDA graphs) leak across ~200 iterations of
# this closed loop and hit RESOURCE_EXHAUSTED; disabling them is XLA's own
# suggested fix. Scoped to this script, not `oim/__init__.py`: doing it
# globally breaks the Warp backend elsewhere.
os.environ["XLA_FLAGS"] = (
    os.environ.get("XLA_FLAGS", "") + " --xla_gpu_enable_command_buffer="
)


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
        return np.asarray(obs.center) + np.asarray(
            rotate(obs.angle, jnp.asarray(corners))
        )
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
    """The object's footprint at a given pose, in world coordinates."""
    return np.asarray(pose[:2]) + np.asarray(
        rotate(float(pose[2]), jnp.asarray(verts))
    )


def _draw_scene(ax, scenario: Scenario, verts: np.ndarray) -> None:  # noqa: ANN001
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


def _base_name(scenario_name: str, method: str) -> str:
    """`pusht2d_{scenario}_{method}_{timestamp}`, no extension.

    Same scheme `examples/pusht.py` uses
    (`pusht3d_{robot}_{method}_{timestamp}`), computed once and reused for
    every output of one run (plot/gif/results), so they pair up under the
    same timestamp instead of each getting its own.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"pusht2d_{scenario_name}_{method}_{timestamp}"


def plot_run(
    task: PushT2D, scenario: Scenario, log: dict, path: str, stride: int = 5
) -> None:
    """Draw the object's swept footprint, the robot path, and diagnostics."""
    # Imported here, not at module scope: the backend must be selected
    # before pyplot is first imported, and keeping both local means a
    # headless run with --no-plot never touches matplotlib at all.
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    fig, (ax, ax_r) = plt.subplots(
        1, 2, figsize=(13, 5.5), gridspec_kw={"width_ratios": [1.4, 1]}
    )
    verts = np.asarray(task.footprint.vertices)
    _draw_scene(ax, scenario, verts)

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


def save_animation(
    task: PushT2D, scenario: Scenario, log: dict, path: str, fps: int = 15
) -> None:
    """Write an animated gif of the run.

    Worth having over the static plot: a swept-footprint figure shows
    *where* the object went but not *when* it stalled, reversed, or got
    shoved sideways by a contact that broke -- which is usually the thing
    you are trying to see.
    """
    import matplotlib  # noqa: PLC0415

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415
    from matplotlib import animation  # noqa: PLC0415

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    verts = np.asarray(task.footprint.vertices)
    _draw_scene(ax, scenario, verts)

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


def main() -> None:
    """Parse arguments, run the 2D loop, and optionally plot."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--env",
        choices=list_scenarios(),
        default=DEFAULT_SCENARIO,
        help=f"Push scenario (default: {DEFAULT_SCENARIO}).",
    )
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-admm", type=int, default=6)
    p.add_argument("--samples", type=int, default=64)
    p.add_argument("--horizon", type=int, default=15)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--contact-action",
        action="store_true",
        help="Have the object block decide a contact action [p, f_n, f_t] "
        "and derive the wrench from it, instead of sampling the wrench "
        "directly. Use to A/B the two parameterizations.",
    )
    p.add_argument(
        "--no-relocate",
        action="store_true",
        help="Disable the global contact-point search, leaving only local "
        "drift along the current face.",
    )
    p.add_argument(
        "--no-obstacles", action="store_true", help="Strip the obstacles."
    )
    p.add_argument(
        "--no-jit",
        action="store_true",
        help="Disable jit everywhere so the physics and ADMM math can be "
        "stepped in a debugger.",
    )
    p.add_argument("--animate", action="store_true", help="Also write a gif.")
    p.add_argument("--no-plot", action="store_true")
    args = p.parse_args()

    scenario = build_scenario(args.env)
    print(f"scenario: {scenario.name} -- {scenario.description}")

    task = PushT2D(
        footprint=scenario.footprint,
        goal=scenario.goal,
        obstacles=None if args.no_obstacles else scenario.obstacles,
        contact_actions=args.contact_action,
        relocate_contact=not args.no_relocate,
    )
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
        f"object block: {block_kind}  "
        f"(action dim {task.object_action_dim}, "
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

    base = _base_name(scenario.name, "admm")
    save_run_results(
        os.path.join(ROOT, "results"),
        base,
        hyperparameters=dict(
            env=scenario.name,
            steps=args.steps,
            n_admm=args.n_admm,
            samples=args.samples,
            horizon=args.horizon,
            rho=args.rho,
            gamma=args.gamma,
            seed=args.seed,
            contact_action=args.contact_action,
            relocate_contact=not args.no_relocate,
        ),
        log=log,
    )

    if not args.no_plot:
        out_dir = os.path.join(ROOT, "recordings")
        os.makedirs(out_dir, exist_ok=True)
        plot_run(task, scenario, log, os.path.join(out_dir, f"{base}.png"))
        if args.animate:
            save_animation(
                task, scenario, log, os.path.join(out_dir, f"{base}.gif")
            )


if __name__ == "__main__":
    main()
