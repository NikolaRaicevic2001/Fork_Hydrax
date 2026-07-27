import argparse
import time
from copy import deepcopy

import jax.numpy as jnp
import mujoco
import numpy as np

from hydrax.admm.admm import ADMMController, ADMMState
from hydrax.admm.consensus import WrenchConsensus
from hydrax.admm.object_optimizer import ObjectOptimizer
from hydrax.algs.admm_mppi import ADMMMPPI
from hydrax.algs.mppi import MPPI
from hydrax.tasks.pusht_clutter import GOAL, ClutterObjectTask, ClutterRobotTask

"""
ADMM-coordinated object-informed MPPI on the PushT clutter scene: an
object-level MPPI (closed-form quasi-static wrench dynamics) and a
robot-level MPPI (real MJX contact) reach consensus on the contact wrench
every real control step (root_icra.tex Sec05, Algorithm 4).
"""

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument(
    "--warp", action="store_true", help="Use the MjWarp backend."
)
parser.add_argument(
    "--steps", type=int, default=800, help="Real MPC steps to run."
)
parser.add_argument(
    "--horizon", type=int, default=15, help="Consensus horizon H."
)
parser.add_argument(
    "--n-admm", type=int, default=12, help="Max ADMM iters/step."
)
parser.add_argument(
    "--k-object", type=int, default=64, help="Object MPPI samples."
)
parser.add_argument(
    "--k-robot", type=int, default=64, help="Robot MPPI samples."
)
parser.add_argument("--plan-dt", type=float, default=0.05, help="Consensus dt.")
parser.add_argument(
    "--exec-dt", type=float, default=0.002, help="Execution dt."
)
parser.add_argument("--rho", type=float, default=1.0)
parser.add_argument("--max-dual", type=float, default=15.0)
parser.add_argument("--eps-r", type=float, default=1.0)
parser.add_argument("--eps-s", type=float, default=1.0)
parser.add_argument("--record", action="store_true", help="Record an mp4.")
args = parser.parse_args()

impl = "warp" if args.warp else "jax"
H = args.horizon

consensus = WrenchConsensus(horizon=H, rho=args.rho, max_dual=args.max_dual)

object_task = ClutterObjectTask(consensus, dt=args.plan_dt)
object_mppi = MPPI(
    object_task,
    num_samples=args.k_object,
    noise_level=1.0,
    temperature=1.0,
    plan_horizon=H * args.plan_dt,
    spline_type="zero",
    num_knots=H,
)
object_ctrl = ObjectOptimizer(object_task, object_mppi)

robot_task = ClutterRobotTask(consensus, planning_dt=args.plan_dt, impl=impl)
robot_ctrl = ADMMMPPI(
    robot_task,
    num_samples=args.k_robot,
    noise_level=0.12,
    temperature=1.0,
    plan_horizon=H * args.plan_dt,
    spline_type="zero",
    num_knots=H,
)

admm_ctrl = ADMMController(
    object_ctrl,
    robot_ctrl,
    consensus,
    n_admm=args.n_admm,
    eps_r=args.eps_r,
    eps_s=args.eps_s,
)

admm_state = ADMMState(
    object_params=object_mppi.init_params(),
    robot_params=robot_ctrl.init_params(),
    z=jnp.zeros((H, 3)),
    gamma_o=jnp.zeros((H, 3)),
    gamma_r=jnp.zeros((H, 3)),
)

# Fine execution model: same scene, finer timestep and more solver iterations.
exec_model = deepcopy(robot_task.mj_model)
exec_model.opt.timestep = args.exec_dt
exec_model.opt.iterations = 100
exec_model.opt.ls_iterations = 50
exec_data = mujoco.MjData(exec_model)
mujoco.mj_forward(exec_model, exec_data)
n_sub = max(int(round(args.plan_dt / args.exec_dt)), 1)

renderer = None
recorder = None
if args.record:
    from hydrax.utils.video import VideoRecorder

    try:
        renderer = mujoco.Renderer(exec_model, height=480, width=720)
        recorder = VideoRecorder("results/admm_clutter")
        recorder.start()
    except Exception as e:  # noqa: BLE001
        print(f"Warning: rendering disabled ({e})")
        renderer = None

print(f"Running ADMM-MPPI (impl={impl}) for {args.steps} steps...")
t0 = time.time()
for step in range(args.steps):
    object_pose0 = jnp.array(exec_data.qpos[:3])

    robot_data = robot_task.make_data()
    robot_data = robot_data.replace(
        qpos=jnp.array(exec_data.qpos),
        qvel=jnp.array(exec_data.qvel),
        mocap_pos=jnp.array(exec_data.mocap_pos),
        mocap_quat=jnp.array(exec_data.mocap_quat),
    )

    admm_state, u0, info = admm_ctrl.control_step(
        object_pose0, robot_data, admm_state
    )
    u0 = np.asarray(u0)

    for _ in range(n_sub):
        exec_data.ctrl[:] = u0
        mujoco.mj_step(exec_model, exec_data)

    if renderer is not None:
        renderer.update_scene(exec_data)
        recorder.add_frame(renderer.render().tobytes())

    if step % 10 == 0:
        pos_err = float(jnp.linalg.norm(object_pose0[:2] - GOAL[:2]))
        theta_err = float(jnp.abs(object_pose0[2] - GOAL[2]))
        print(
            f"step {step:4d}  pos_err={pos_err:.3f}  "
            f"theta_err={theta_err:.3f}  admm_iters={info['admm_iters']}  "
            f"primal={info['primal_residual']:.3f}"
        )

print(f"Done in {time.time() - t0:.1f}s")
if recorder is not None:
    recorder.stop()
