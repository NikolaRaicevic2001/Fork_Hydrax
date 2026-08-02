"""Hardware closed-loop driver: the real-robot counterpart of `sim3d/run.py`.

Structurally identical to `oim.sim3d.run._run`. The planner (`ADMM.optimize`),
the task cost and the MJX rollouts are reused unchanged -- MJX is still the
planner's internal predictive model on real hardware, exactly as in
simulation; what the real world replaces is only *execution* and *state*:

    sim3d._run                       real3d.run_real
    ----------------------------     --------------------------------------
    mjx_data <- mj_data.qpos/qvel    mjx_data <- interface.read_state()
    mj_data.ctrl = u ; mj_step(...)  interface.send_velocity(u)

Because the planner is a plain jitted JAX function, it is called directly in
this process -- no zerorpc, no separate planner server (those were a
workaround for Isaac Gym's per-process sim context, which MJX does not need).

REAL-TIME MODEL (matches the OI-MPPI/Isaac reference). One `optimize` per
control step, publishing a single velocity that the arm's velocity controller
*holds* until the next step. The arm therefore keeps moving at the last
command during the ~0.3-0.4 s planning time; it does not stop. The only hard
requirement is that the replanning period be >= the optimize time, so pick
`replan_rate` accordingly (MJX xarm6 optimize ~0.36 s -> ~2.5 Hz; the Isaac
MPPI ran at 5 Hz). A safety watchdog that commands zero velocity when planning
stalls belongs on the interface's own concurrent executor (see
`Ros2Interface`), not in this synchronous loop.

The state log uses the exact same keys/schema as `sim3d/run.py`, so a hardware
run and a simulation run compare entry-for-entry -- the sim-to-real validation
this port exists to produce.
"""

from __future__ import annotations

import time
from typing import Any, Dict

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from oim.objects import wrap_angle
from oim.real3d.interface import RobotWorldInterface, SceneAddresses, clamp_velocity
from oim.sim3d.run import _finalize_log, _init_log, _log_step
from oim.tasks.pusht import PushT


def run_real(
    task: PushT,
    ctrl: Any,  # ADMM
    params: Any,
    interface: RobotWorldInterface,
    replan_rate: float = 2.5,
    control_rate: float = 50.0,
    command_mode: str = "hold",
    max_steps: int = 200,
    goal_pos_tol: float = 0.05,
    goal_theta_tol: float = 0.05,
    real_time: bool = False,
    verbose: bool = True,
) -> Dict[str, Any]:
    """Run the push-T ADMM controller against a `RobotWorldInterface`.

    Args:
        task: the `PushT` task, built with `robot="xarm6"`.
        ctrl: the ADMM controller, built against `task` (reuse the exact
            construction from `examples/pusht.py`).
        params: initial policy parameters (`ctrl.init_params(...)`).
        interface: hardware or mock world. `MujocoMockInterface` for laptop
            testing, `Ros2Interface` on the robot.
        replan_rate: replanning frequency (Hz) -- how much sim/world time one
            `optimize` covers. Must be <= 1/optimize time to run in real time.
        control_rate: rate (Hz) at which velocity commands are streamed to the
            arm across each replanning period. Higher than `replan_rate`.
        command_mode: how the plan is turned into commands within a period:
            "hold" streams a single, constant velocity (the plan's value at the
            start of the period) -- what the OI-MPPI/Isaac interface does, and
            what a velocity controller does implicitly between plans.
            "stream" streams the plan's *time-varying* velocity, sampled at
            `control_rate` -- more faithful to the plan when `replan_rate` is
            low, at the cost of running open-loop over the period. (True
            continuous streaming across replans ultimately wants the async
            driver; this synchronous loop streams over one fixed window.)
        max_steps: maximum replanning steps.
        goal_pos_tol, goal_theta_tol: success tolerances.
        real_time: if True, pace command streaming to `control_rate` on the
            wall clock (hardware); if False, let the interface advance itself
            (mock sim), which is deterministic and rate-independent.
        verbose: print per-step progress.

    Returns:
        A log dict with the same schema as `sim3d.run.run_3d_admm`.
    """
    if command_mode not in ("hold", "stream"):
        raise ValueError(f"command_mode must be 'hold' or 'stream', got {command_mode!r}")
    addresses = SceneAddresses.from_model(task.mj_model)
    replan_period = 1.0 / replan_rate
    control_dt = 1.0 / control_rate
    num_ticks = max(1, round(replan_period / control_dt))

    jit_optimize = jax.jit(ctrl.optimize)
    jit_interp = jax.jit(ctrl.interp_func)

    # Assemble the first MJX state, then warm up the JIT before the timed loop
    # (compilation happens on the first one or two calls).
    base_data = task.make_data()
    state0 = interface.read_state()
    mjx_data = _assemble_state(task, base_data, addresses, state0)
    if verbose:
        print("Warming up the controller (JIT compile)...")
    t_jit = time.time()
    params, _ = jit_optimize(mjx_data, params)
    params, _ = jit_optimize(mjx_data, params)
    _ = jit_interp(jnp.arange(num_ticks) * control_dt, params.tk,
                   params.mean[None, ...])
    jax.block_until_ready(params)
    if verbose:
        print(f"JIT done in {time.time() - t_jit:.2f}s; replan {replan_rate:.1f} Hz, "
              f"stream {control_rate:.0f} Hz ({num_ticks} cmds/period, {command_mode})")

    log = _init_log(task, mjx_data, mjx_data, show_plans=False)
    reached = False

    for step in range(max_steps):
        # (A) INPUT seam: read the world, build the planner's MJX state.
        world = interface.read_state()
        mjx_data = _assemble_state(task, base_data, addresses, world)

        # (B) Plan (one optimize; MJX rollouts still happen inside it).
        t0 = time.perf_counter()
        params, _ = jit_optimize(mjx_data, params)
        jax.block_until_ready(params)
        compute_time = time.perf_counter() - t0
        log["compute_time"].append(compute_time)

        # (C) Sample the plan across the upcoming period at the control rate.
        sample_times = jnp.arange(num_ticks) * control_dt + world.time
        plan_samples = np.asarray(
            jit_interp(sample_times, params.tk, params.mean[None, ...])
        )[0]  # (num_ticks, nu)

        # (D) OUTPUT seam: stream the commands to the arm (or mock sim). In
        # "hold" every tick sends the same first velocity; in "stream" each
        # tick sends the plan's value for that time.
        t_stream = time.perf_counter()
        applied = np.empty_like(plan_samples)
        for i in range(num_ticks):
            velocity = plan_samples[0] if command_mode == "hold" else plan_samples[i]
            velocity = clamp_velocity(velocity)
            applied[i] = velocity
            interface.send_velocity(velocity)
            if real_time:
                _sleep_until(t_stream + (i + 1) * control_dt)

        # (E) Log and check the goal. `_log_step` reads a MuJoCo-data-like
        # object; the forwarded mjx_data satisfies that (qpos, qvel, time,
        # site_xpos all present), so the sim logger is reused as-is.
        block_pose = _log_step(log, task, mjx_data, params, applied)
        goal = np.asarray(task.goal)
        pos_err = float(np.linalg.norm(block_pose[:2] - goal[:2]))
        theta_err = float(abs(float(wrap_angle(block_pose[2] - goal[2]))))
        log["pos_err"].append(pos_err)
        log["theta_err"].append(theta_err)
        if verbose and step % 10 == 0:
            print(f"step {step:4d}  pos_err={pos_err:.4f}  "
                  f"theta_err={theta_err:.4f}  "
                  f"primal={log['primal_residual'][-1]:.3f}  "
                  f"plan={compute_time * 1e3:.0f}ms")
        if pos_err < goal_pos_tol and theta_err < goal_theta_tol:
            reached = True
            if verbose:
                print(f"goal reached at step {step}")
            break

    # Stop the arm before returning.
    interface.send_velocity(np.zeros(len(world.arm_qpos)))
    return _finalize_log(log, task, reached, show_plans=False)


def _sleep_until(deadline: float) -> None:
    """Sleep until a wall-clock deadline (seconds, perf_counter timebase)."""
    remaining = deadline - time.perf_counter()
    if remaining > 0:
        time.sleep(remaining)


def _assemble_state(
    task: PushT,
    base_data: mjx.Data,
    adr: SceneAddresses,
    world: Any,  # WorldState
) -> mjx.Data:
    """Inject measured arm + object state into a full MJX state.

    The planner needs a `mjx.Data` for the *whole* scene (arm joints + the
    block's SE(2) joints); hardware only measures the arm (encoders) and the
    object (FoundationPose). The static obstacles are already baked into the
    model, so we only write the two moving parts into their qpos/qvel slots
    -- looked up by joint name in `SceneAddresses`, never assumed to be a
    fixed slice -- and run forward kinematics so `site_xpos` (the stick tip)
    is populated for the cost and the logger.
    """
    nq = task.mj_model.nq
    nv = task.mj_model.nv
    qpos = np.asarray(base_data.qpos).copy()
    qvel = np.zeros(nv)

    qpos[adr.arm_qpos_adr] = world.arm_qpos
    qpos[adr.block_qpos_adr] = world.object_se2
    qvel[adr.arm_dof_adr] = world.arm_qvel
    # Block twist feeds realized_consensus (the "twist" A^r estimator):
    # w = wrench_limit * qvel[block_dofs].
    qvel[adr.block_dof_adr] = world.object_twist

    assert qpos.shape[0] == nq, (qpos.shape, nq)
    mjx_data = base_data.replace(
        qpos=jnp.asarray(qpos),
        qvel=jnp.asarray(qvel),
        time=float(world.time),
    )
    return mjx.forward(task.model, mjx_data)
