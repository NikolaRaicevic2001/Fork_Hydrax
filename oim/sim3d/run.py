"""Headless closed-loop driver for the MJX/MuJoCo world, with ADMM logging.

The 3D counterpart of `oim.sim2d.run.run_2d`: steps the real `mujoco.MjData`
and the MJX planning model in lockstep, returning the same kind of log dict
`run_2d` does (trajectories, wrenches, primal/dual residuals, goal errors),
for `oim.tasks.pusht.PushT` under either `robot` embodiment. Headless -- no
viewer, no recording -- reuses `oim.sim3d.deterministic.run_interactive`'s
stepping logic, but that function is generic over any controller/task and
has no way to return a log, which is what this fills in for the ADMM+PushT
case specifically.
"""

from typing import Any, Dict

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from oim.algs.admm import ADMM
from oim.objects import wrap_angle
from oim.tasks.pusht import PushT


def run_3d_admm(
    task: PushT,
    ctrl: ADMM,
    params: Any,
    mj_model: mujoco.MjModel,
    mj_data: mujoco.MjData,
    frequency: float,
    max_steps: int = 200,
    goal_pos_tol: float = 0.06,
    goal_theta_tol: float = 0.10,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the MJX closed loop under ADMM and return a log, headless.

    Args:
        task: The `PushT` task (`robot="point"` or `"xarm6"`).
        ctrl: The ADMM controller, built against `task`.
        params: Its initial policy parameters.
        mj_model: The (fine-timestep) execution model.
        mj_data: Its initial state.
        frequency: Replanning frequency (Hz).
        max_steps: Maximum control steps.
        goal_pos_tol: Positional tolerance for declaring success.
        goal_theta_tol: Angular tolerance for declaring success.
        verbose: Whether to print progress.

    Returns:
        A dict with the block/pusher trajectories, per-step wrenches,
        primal/dual residuals, goal errors, and whether the goal was reached.
    """
    replan_period = 1.0 / frequency
    sim_steps_per_replan = max(int(replan_period / mj_model.opt.timestep), 1)

    mjx_data = task.make_data()
    mjx_data = mjx_data.replace(
        qpos=mj_data.qpos,
        qvel=mj_data.qvel,
        mocap_pos=mj_data.mocap_pos,
        mocap_quat=mj_data.mocap_quat,
    )
    # Populate site_xpos etc. before the first log entry reads them
    # (a freshly made mjx.Data hasn't run forward kinematics yet).
    mjx_data = mjx.forward(task.model, mjx_data)
    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp_func = jax.jit(ctrl.interp_func)

    log: Dict[str, Any] = {
        "block_pose": [np.array(task._block_pose(mjx_data))],
        "pusher_pos": [np.array(task._pusher_pos(mjx_data))],
        "wrench": [],
        "primal_residual": [],
        "dual_residual": [],
        "rho": [],
        "pos_err": [],
        "theta_err": [],
    }
    reached = False

    for step in range(max_steps):
        mjx_data = mjx_data.replace(
            qpos=jnp.array(mj_data.qpos),
            qvel=jnp.array(mj_data.qvel),
            mocap_pos=jnp.array(mj_data.mocap_pos),
            mocap_quat=jnp.array(mj_data.mocap_quat),
            time=mj_data.time,
        )
        params, _ = jit_optimize(mjx_data, params)

        tq = (
            jnp.arange(sim_steps_per_replan) * mj_model.opt.timestep
            + mj_data.time
        )
        tk = params.tk
        knots = params.mean[None, ...]
        us = np.asarray(jit_interp_func(tq, tk, knots))[0]
        for i in range(sim_steps_per_replan):
            mj_data.ctrl[:] = us[i]
            mujoco.mj_step(mj_model, mj_data)

        block_pose = np.array(task._block_pose(mj_data))
        log["block_pose"].append(block_pose)
        log["pusher_pos"].append(np.array(task._pusher_pos(mj_data)))
        log["wrench"].append(np.array(task.realized_consensus(mj_data)))
        log["primal_residual"].append(float(params.primal_residual))
        log["dual_residual"].append(float(params.dual_residual))
        log["rho"].append(float(params.rho))

        goal = np.asarray(task.goal)
        pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
        theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
        log["pos_err"].append(pos_err)
        log["theta_err"].append(theta_err)
        if verbose and step % 10 == 0:
            print(
                f"step {step:4d}  pos_err={pos_err:.4f}  "
                f"theta_err={theta_err:.4f}  "
                f"primal={log['primal_residual'][-1]:.3f}  "
                f"dual={log['dual_residual'][-1]:.3f}  "
                f"rho={log['rho'][-1]:.2f}"
            )
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

    log["reached"] = reached
    log["block_pose"] = np.array(log["block_pose"])
    log["pusher_pos"] = np.array(log["pusher_pos"])
    log["wrench"] = (
        np.array(log["wrench"]) if log["wrench"] else np.zeros((0, 3))
    )
    return log
