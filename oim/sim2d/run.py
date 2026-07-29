"""Closed-loop driver for the 2D world.

The 2D analogue of `oim.sim3d.deterministic.run_interactive`, minus the
MuJoCo viewer: it steps the analytic engine, logs everything the ADMM layer
produces, and hands back arrays for plotting. Because the whole loop is
plain Python around jitted calls, you can drop a breakpoint anywhere, or
wrap a call in `jax.disable_jit()` to step into the physics itself.
"""

from typing import Any, Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np

from oim.alg_base import SamplingBasedController
from oim.algs import ADMM, MPPI, WrenchConsensus, make_object_shim
from oim.objects import wrap_angle
from oim.sim2d.task import PushT2D


def build_admm_2d(
    task: PushT2D,
    horizon: int = 15,
    num_samples: int = 64,
    robot_knots: int = 4,
    n_admm: int = 8,
    rho: float = 10.0,
    gamma: float = 0.1,
    eps_r: float = 0.5,
    eps_s: float = 0.5,
    seed: int = 0,
    robot_optimizer: Optional[SamplingBasedController] = None,
    object_optimizer: Optional[SamplingBasedController] = None,
) -> Tuple[ADMM, Any]:
    """Assemble an `ADMM` controller wired to the 2D rollout backend.

    The defaults mirror `examples/pusht.py`'s ADMM branch so 2D and MJX runs
    are comparable, except for the residual tolerances: `eps_r`/`eps_s` are
    0.5 rather than 0.05 because the residual is a Frobenius norm over
    `(2H, 3)` normalized entries, and 0.05 is not reachable at H = 15 -- the
    early exit would never fire and every step would burn all `n_admm`
    iterations.

    Args:
        task: The 2D task.
        horizon: Consensus horizon H, in steps of `task.dt`.
        num_samples: Rollouts per sub-optimizer.
        robot_knots: Spline knots for the robot block.
        n_admm: Maximum ADMM iterations per control step.
        rho: Initial ADMM penalty weight.
        gamma: Proximal weight.
        eps_r: Primal residual tolerance for early exit.
        eps_s: Dual residual tolerance for early exit.
        seed: Random seed.
        robot_optimizer: Override the robot-block sampler.
        object_optimizer: Override the object-block sampler.

    Returns:
        The controller and its initial policy parameters.
    """
    plan_horizon = horizon * task.dt

    if robot_optimizer is None:
        robot_optimizer = MPPI(
            task,
            num_samples=num_samples,
            noise_level=0.2,
            temperature=0.5,
            plan_horizon=plan_horizon,
            spline_type="linear",
            num_knots=robot_knots,
            seed=seed,
        )
    if object_optimizer is None:
        object_optimizer = MPPI(
            make_object_shim(task, dt=task.dt),
            num_samples=num_samples,
            noise_level=0.0,  # the task owns the proposal; see below
            temperature=1.0,
            plan_horizon=plan_horizon,
            spline_type="zero",
            num_knots=horizon,
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
        n_admm=n_admm,
        eps_r=eps_r,
        eps_s=eps_s,
        proximal_weight=gamma,
        rho_init=rho,
        noise_min=0.0,
        noise_kappa=0.0,
        noise_max=0.0,
        rollout=task.rollout,
        debug_print=False,
    )
    return ctrl, ctrl.init_params(seed=seed)


def run_2d(
    task: PushT2D,
    ctrl: ADMM,
    params: Any,
    object_pose0: Tuple[float, float, float] = (0.0, 0.0, 0.0),
    robot_pos0: Tuple[float, float] = (-0.1, -0.12),
    max_steps: int = 200,
    goal_pos_tol: float = 0.06,
    goal_theta_tol: float = 0.10,
    verbose: bool = True,
    jit: bool = True,
) -> Dict[str, Any]:
    """Run the 2D closed loop and return a log.

    Args:
        task: The 2D task.
        ctrl: The ADMM controller from `build_admm_2d`.
        params: Its initial policy parameters.
        object_pose0: Initial object pose.
        robot_pos0: Initial robot position.
        max_steps: Maximum control steps.
        goal_pos_tol: Positional tolerance for declaring success.
        goal_theta_tol: Angular tolerance for declaring success.
        verbose: Whether to print progress.
        jit: Whether to jit the optimize call. Set False to step through
            the physics and the ADMM math in a debugger.

    Returns:
        A dict with the object/robot trajectories, per-step wrenches and
        residuals, and whether the goal was reached.
    """
    optimize = jax.jit(ctrl.optimize) if jit else ctrl.optimize
    get_action = jax.jit(ctrl.get_action) if jit else ctrl.get_action

    state = task.make_data(object_pose=object_pose0, robot_pos=robot_pos0)
    log: Dict[str, Any] = {
        "object_pose": [np.asarray(state.object_pose)],
        "robot_pos": [np.asarray(state.robot_pos)],
        "wrench": [],
        "w_obj": [],
        "primal_residual": [],
        "rho": [],
    }
    reached = False

    for step in range(max_steps):
        params, _ = optimize(state, params)
        u = get_action(params, state.time)
        u = jnp.clip(u, task.u_min, task.u_max)
        state = task.rollout.step(task.sim_model, state, u)

        log["object_pose"].append(np.asarray(state.object_pose))
        log["robot_pos"].append(np.asarray(state.robot_pos))
        log["wrench"].append(np.asarray(state.wrench))
        log["w_obj"].append(np.asarray(params.z))
        log["primal_residual"].append(float(params.primal_residual))
        log["rho"].append(float(params.rho))

        pos_err = float(jnp.linalg.norm(state.object_pose[:2] - task.goal[:2]))
        theta_err = float(
            jnp.abs(wrap_angle(state.object_pose[2] - task.goal[2]))
        )
        if verbose and step % 10 == 0:
            print(
                f"step {step:4d}  pos_err={pos_err:.4f}  "
                f"theta_err={theta_err:.4f}  "
                f"|w_rob|={float(jnp.linalg.norm(state.wrench)):.3f}  "
                f"primal={log['primal_residual'][-1]:.3f}  "
                f"rho={log['rho'][-1]:.2f}"
            )
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

    log["reached"] = reached
    log["object_pose"] = np.array(log["object_pose"])
    log["robot_pos"] = np.array(log["robot_pos"])
    log["wrench"] = (
        np.array(log["wrench"]) if log["wrench"] else np.zeros((0, 3))
    )
    return log
