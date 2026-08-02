"""Entry point: run the push-T ADMM controller on the real xArm6 (or a mock).

The hardware sibling of `examples/pusht.py`'s `--robot xarm6 admm --headless`
path. The task and the ADMM controller are built exactly as there -- same
scene, same weights, same hyperparameters -- so any difference in behaviour
is the sim-to-real gap, not a different planner. Only the execution driver
changes: `oim.real3d.run_real` instead of `oim.sim3d.run.run_3d_admm`.

    # Laptop / dev: drive a MuJoCo sim through the hardware interface.
    #   Validates the whole loop with no robot and no ROS.
    python examples/pusht_real.py --mock --steps 200

    # Robot (at the lab, with the arm + FoundationPose + velocity controller):
    python examples/pusht_real.py --steps 400

The states/metrics JSON is written with the same schema as the simulation
run, so `oim.utils.metrics` / a diff of the two files gives the sim-to-real
comparison directly.
"""

import argparse
import math
import os
import warnings
from copy import deepcopy

# Cosmetic warnings during CPU JAX tracing / MuJoCo model compile. The results
# are unaffected (the saved states contain no NaNs); filtered here at the entry
# point only so the closed-loop log stays readable.
warnings.filterwarnings("ignore", message="overflow encountered in cast")
warnings.filterwarnings("ignore", message=".*coplanar face.*")

import mujoco

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
from oim.real3d.interface import MujocoMockInterface
from oim.real3d.run_real import run_real
from oim.tasks.pusht import PushT
from oim.utils.results import RunName, save_run_metrics, save_run_states

PLAN_DT = 0.05      # planner timestep (matches examples/pusht.py)
HORIZON = 15        # consensus horizon H, in steps of PLAN_DT
EXEC_TIMESTEP = 0.002  # fine execution timestep for the mock sim
# Starting arm configuration (deg): stick tip near the block, from the reach
# sweep in models/xarm6_pusht_clutter/verify_reach.py. Kept in sync with the
# same constant in examples/pusht.py.
XARM6_START_QPOS_DEG = [-15.43, 100.0, -185.36, 0.0, 60.0]


def build_sub_optimizer(name, task, *, plan_horizon, num_knots, spline, seed):
    """Identical to examples/pusht.py::build_sub_optimizer (kept in sync)."""
    common = dict(
        plan_horizon=plan_horizon,
        spline_type=spline,
        num_knots=num_knots,
        seed=seed,
    )
    if name == "mppi":
        return MPPI(task, num_samples=64, noise_level=0.5, temperature=0.5, **common)
    if name == "cem":
        return CEM(task, num_samples=64, num_elites=8, sigma_start=0.5,
                   sigma_min=0.1, **common)
    if name == "ps":
        return PredictiveSampling(task, num_samples=64, noise_level=0.5, **common)
    if name == "cbo":
        return CBO(task, num_samples=64, initial_noise_level=0.5, temperature=0.5,
                   consensus_weight=1.0, noise_weight=1.0, step_size=0.1, **common)
    raise ValueError(f"unknown sub-optimizer '{name}'")


def build_controller(args):
    """Build the xArm6 PushT task + ADMM controller, exactly as in pusht.py."""
    task = PushT(
        impl="warp" if args.warp else "jax",  # --warp: MuJoCo Warp rollout backend
        clutter=True,
        planning_dt=PLAN_DT,
        robot="xarm6",
        consensus_source="twist",  # only valid estimator for an articulated arm
    )
    consensus = WrenchConsensus(
        max_dual=2.0 * float(task.consensus_scale()[0]),
        scale=task.consensus_scale(),
    )
    robot_optimizer = build_sub_optimizer(
        args.robot_opt, task, plan_horizon=HORIZON * PLAN_DT,
        num_knots=4, spline="linear", seed=args.seed,
    )
    object_optimizer = build_sub_optimizer(
        args.object_opt, make_object_shim(task, dt=PLAN_DT),
        plan_horizon=HORIZON * PLAN_DT, num_knots=HORIZON, spline="zero",
        seed=args.seed,
    )
    ctrl = ADMM(
        task, robot_optimizer, object_optimizer, consensus,
        n_admm=args.n_admm, eps_r=0.5, eps_s=0.5,
        proximal_weight=args.gamma, rho_init=args.rho,
        # Hyperparameters copied verbatim from examples/pusht.py's ADMM branch
        # (residual-relative noise anneal + damped consensus update).
        noise_min=0.0, noise_kappa=0.3, noise_max=0.3,
        consensus_relax=0.3,
    )
    return task, ctrl


def build_mock_interface(task, control_rate):
    """A MuJoCo sim behind the hardware interface, for laptop testing.

    Each `send_velocity` applies the commanded velocity and advances the sim by
    one control tick (1/control_rate). `run_real` calls it `num_ticks` times per
    replanning period, so the sim advances exactly one period per plan.
    """
    mj_model = deepcopy(task.mj_model)
    mj_model.opt.timestep = EXEC_TIMESTEP
    mj_model.opt.iterations = 100
    mj_model.opt.ls_iterations = 50
    mj_data = mujoco.MjData(mj_model)
    # Start pose: arm near the block (reach sweep), block at the origin.
    mj_data.qpos[:5] = [math.radians(q) for q in XARM6_START_QPOS_DEG]
    mj_data.qpos[5:8] = [0.0, 0.0, 0.0]
    sim_steps_per_send = max(1, round((1.0 / control_rate) / EXEC_TIMESTEP))
    return MujocoMockInterface(mj_model, mj_data, sim_steps_per_send)


def build_real_interface(task, velocity_topic):
    """The real ROS2 <-> xArm6 bridge. Import is lazy so --mock needs no ROS.

    Defaults (frames, joint naming, watchdog) are set from the OI-MPPI
    reference in Ros2Interface.__init__; only the command topic is passed
    through here so `--dry-run` can redirect it to a non-motor topic.
    """
    from oim.real3d.interface import Ros2Interface  # noqa: PLC0415

    return Ros2Interface(velocity_command_topic=velocity_topic)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mock", action="store_true",
                   help="drive a MuJoCo sim instead of the real robot")
    p.add_argument("--steps", type=int, default=200, help="max control steps")
    p.add_argument("--replan-rate", type=float, default=2.5,
                   help="replanning frequency (Hz); must be <= 1/optimize time")
    p.add_argument("--control-rate", type=float, default=50.0,
                   help="velocity command streaming rate (Hz)")
    p.add_argument("--command-mode", default="hold", choices=["hold", "stream"],
                   help="hold a constant velocity per period, or stream the plan")
    p.add_argument("--warp", action="store_true",
                   help="use the MuJoCo Warp rollout backend (speed A/B)")
    p.add_argument("--velocity-topic", default="velocity_controller/commands",
                   help="topic to publish arm velocity commands to")
    p.add_argument("--dry-run", action="store_true",
                   help="publish to <velocity-topic>_nominal (no motion): plan "
                        "and watch in RViz without commanding the arm")
    p.add_argument("--socket", action="store_true",
                   help="two-process fallback: talk to oim.real3d.ros_bridge "
                        "over a socket instead of importing rclpy here")
    p.add_argument("--bridge-host", default="127.0.0.1")
    p.add_argument("--bridge-port", type=int, default=5599)
    p.add_argument("--n-admm", type=int, default=8)
    p.add_argument("--rho", type=float, default=10.0)
    p.add_argument("--gamma", type=float, default=0.1)
    p.add_argument("--robot-opt", default="mppi", choices=["mppi", "cem", "ps", "cbo"])
    p.add_argument("--object-opt", default="mppi", choices=["mppi", "cem", "ps", "cbo"])
    p.add_argument("--seed", type=int, default=5)
    args = p.parse_args()

    task, ctrl = build_controller(args)

    if args.mock:
        interface = build_mock_interface(task, args.control_rate)
        real_time = False
    elif args.socket:
        # Two-process fallback: the ROS I/O (and --dry-run / --velocity-topic)
        # live in oim.real3d.ros_bridge, running in the ROS env.
        from oim.real3d.interface import SocketInterface  # noqa: PLC0415
        interface = SocketInterface(args.bridge_host, args.bridge_port)
        real_time = True
    else:
        # Dry run: publish to a *_nominal topic that is not wired to the
        # motors, so perception + planning can be checked in RViz without
        # moving the arm (mirrors OI-MPPI's enable_velocity_commands=false).
        topic = args.velocity_topic + ("_nominal" if args.dry_run else "")
        interface = build_real_interface(task, topic)
        real_time = True

    try:
        log = run_real(
            task, ctrl, ctrl.init_params(seed=args.seed), interface,
            replan_rate=args.replan_rate,
            control_rate=args.control_rate,
            command_mode=args.command_mode,
            max_steps=args.steps,
            real_time=real_time,
        )
    finally:
        interface.close()

    # Same naming/schema as the sim run, so the two logs compare directly.
    results_dir = os.path.join(ROOT, "results")
    os.makedirs(results_dir, exist_ok=True)
    variant = "xarm6_mock" if args.mock else "xarm6_real"
    name = RunName("pusht3d", variant, "admm")
    save_run_metrics(
        results_dir, name,
        hyperparameters=dict(
            robot="xarm6", mock=args.mock, steps=args.steps,
            n_admm=args.n_admm, robot_opt=args.robot_opt,
            object_opt=args.object_opt, rho=args.rho, gamma=args.gamma,
            seed=args.seed, replan_rate=args.replan_rate,
            control_rate=args.control_rate, command_mode=args.command_mode,
            warp=args.warp,
        ),
        log=log,
    )
    save_run_states(
        results_dir, name, task, log,
        extra_static=dict(
            robot="xarm6", mock=args.mock,
            qpos_size=int(task.mj_model.nq),
            qvel_size=int(task.mj_model.nv),
            block_qpos_adr=task.block_qpos_adr,
            block_dof_adr=task.block_dofs,
        ),
    )
    print(f"saved results to {results_dir} ({name()})")


if __name__ == "__main__":
    main()
