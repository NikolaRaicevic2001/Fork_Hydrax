from typing import Dict, Optional

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from hydrax import ROOT
from hydrax.objects import (
    Box,
    Circle,
    ObstacleField,
    PlanarPushingObject,
    Polygon,
    se2_distance_sq,
    t_shape_footprint,
)
from hydrax.task_base import ConsensusTask, Task

# Goal pose for the cluttered variant (world-frame SE(2)), matching the
# `goal` mocap body in models/pusht_clutter/pusht_clutter.xml.
GOAL = jnp.array([0.50, 0.48, jnp.pi / 4])

# Static obstacles, matching the obstacle geoms in the same MJCF.
CLUTTER_OBSTACLES = ObstacleField(
    [
        Circle(center=[0.08, 0.32], radius=0.04),
        Box(center=[0.38, 0.10], half_extents=[0.04, 0.035], angle=0.25),
        Polygon(jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])),
    ]
)


class PushT(Task, ConsensusTask):
    """Push a T-shaped block to a desired pose, optionally through clutter.

    With `clutter=False` (default), loads the plain `models/pusht` scene and
    supports ordinary sampling-based MPC (`running_cost`/`terminal_cost`).

    With `clutter=True`, loads `models/pusht_clutter` (static obstacles, and
    a model whose joint friction is tuned to match the analytic limit-surface
    object model) and additionally implements `ConsensusTask`, so it can be
    driven by `hydrax.algs.admm.ADMM`. The object-level subproblem is
    delegated to `hydrax.objects.PlanarPushingObject`.
    """

    def __init__(
        self,
        impl: str = "jax",
        clutter: bool = False,
        planning_dt: Optional[float] = None,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            impl: The backend implementation for rollouts ("jax" or "warp").
            clutter: Whether to load the cluttered scene (with obstacles)
                and enable the ADMM `ConsensusTask` methods.
            planning_dt: If given, overrides the model's simulation timestep.
                Used to run the planner at a coarser rate than execution.
        """
        self.clutter = clutter
        scene = "pusht_clutter/scene.xml" if clutter else "pusht/scene.xml"
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/" + scene)
        if planning_dt is not None:
            mj_model.opt.timestep = planning_dt
        super().__init__(mj_model, trace_sites=["pusher"], impl=impl)

        # Sensor ids (defined identically in both scenes).
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

        if clutter:
            pusher_x_dof = mj_model.joint("root_x").dofadr[0]
            pusher_y_dof = mj_model.joint("root_y").dofadr[0]
            self.pusher_dofs = jnp.array([pusher_x_dof, pusher_y_dof])

            # Analytic object-level subproblem. mu/mass are chosen so that
            # the friction-cone limit mu*m*g equals the block joints'
            # `frictionloss` in the MJCF -- the analytic model and the
            # simulated model then describe the same physics.
            self.object_model = PlanarPushingObject(
                dt=self.dt,
                goal=GOAL,
                footprint=t_shape_footprint(),
                obstacles=CLUTTER_OBSTACLES,
                mu=0.4,
                mass=2.0,
                limit_surface_radius=0.06,
            )

            # Robot-level cost weights (paper eq. 20), minus the tilt term:
            # this is a planar pusher with no end-effector orientation DOF.
            self.r_r = 0.05
            self.w_ee, self.r0 = 20.0, 0.05
            self.w_align, self.gamma0 = 5.0, jnp.cos(jnp.pi / 6)
            self.q_pos, self.q_theta = 40.0, 10.0
            self.qf_pos, self.qf_theta = 500.0, 150.0
            self.goal = GOAL

    # ------------------------------------------------------------------
    # Plain (non-ADMM) sampling-based MPC interface
    # ------------------------------------------------------------------

    def _get_position_err(self, state: mjx.Data) -> jax.Array:
        """Position of the block relative to the target position."""
        sensor_adr = self.model.sensor_adr[self.block_position_sensor]
        return state.sensordata[sensor_adr : sensor_adr + 3]

    def _get_orientation_err(self, state: mjx.Data) -> jax.Array:
        """Orientation of the block relative to the target orientation."""
        sensor_adr = self.model.sensor_adr[self.block_orientation_sensor]
        block_quat = state.sensordata[sensor_adr : sensor_adr + 4]
        goal_quat = jnp.array([1.0, 0.0, 0.0, 0.0])
        return mjx._src.math.quat_sub(block_quat, goal_quat)

    def _close_to_block_err(self, state: mjx.Data) -> jax.Array:
        """Position of the pusher block relative to the block."""
        block_pos = state.qpos[:2]
        pusher_pos = state.qpos[3:] + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) for plain (non-ADMM) MPC."""
        position_cost = jnp.sum(jnp.square(self._get_position_err(state)))
        orientation_cost = jnp.sum(jnp.square(self._get_orientation_err(state)))
        close_cost = jnp.sum(jnp.square(self._close_to_block_err(state)))
        return position_cost + orientation_cost + 0.01 * close_cost

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ℓ_T(x_T) for plain (non-ADMM) MPC."""
        return self.running_cost(state, jnp.zeros(self.model.nu))

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Randomize the level of friction."""
        n_geoms = self.model.geom_friction.shape[0]
        multiplier = jax.random.uniform(rng, (n_geoms,), minval=0.1, maxval=2.0)
        new_frictions = self.model.geom_friction.at[:, 0].set(
            self.model.geom_friction[:, 0] * multiplier
        )
        return {"geom_friction": new_frictions}

    def make_data(self) -> mjx.Data:
        """Create a new state object with extra constraints allocated."""
        if self.clutter:
            # Enough contact slots for the pusher, block, and 3 obstacles;
            # the default is too small and silently drops contacts.
            return super().make_data(nconmax=128, naconmax=1024)
        return super().make_data(nconmax=6000)

    # ------------------------------------------------------------------
    # ConsensusTask (ADMM) interface -- only meaningful when clutter=True
    # ------------------------------------------------------------------

    def _block_pose(self, state: mjx.Data) -> jax.Array:
        return state.qpos[:3]

    def _pusher_pos(self, state: mjx.Data) -> jax.Array:
        return state.qpos[3:5]

    @property
    def consensus_dim(self) -> int:
        """The consensus variable is the planar wrench [f_x, f_y, tau]."""
        return 3

    def consensus_scale(self) -> jax.Array:
        """Characteristic wrench magnitude: the friction-cone limit.

        Used by `WrenchConsensus` to normalize the ADMM penalty/residuals.
        """
        return self.object_model.wrench_limit

    def object_action_scale(self) -> jax.Array:
        """Map a unit sample from the object optimizer to a physical wrench."""
        return self.object_model.action_scale

    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Quasi-static limit-surface dynamics (paper eq. 5)."""
        return self.object_model.step(obj_state, w)

    def object_running_cost(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """Object stage cost: goal tracking + clearance + effort (eq. 18)."""
        return self.object_model.running_cost(obj_state, w)

    def object_terminal_cost(self, obj_state: jax.Array) -> jax.Array:
        """Object terminal cost, heavier goal tracking only."""
        return self.object_model.terminal_cost(obj_state)

    def object_state_from_robot(self, state: mjx.Data) -> jax.Array:
        """Extract the object's SE(2) pose from the combined robot state."""
        return self._block_pose(state)

    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """A^r: the wrench the pusher applies to the object (paper eq. 23).

        `qfrc_constraint` at the pusher's DOFs is the constraint force acting
        *on the pusher*; by Newton's third law its negation is the force the
        pusher applies to the object. Expressed in the world frame about the
        block's pose origin -- the same frame and reference point the object
        model integrates, and the same units, so both ADMM blocks report the
        identical physical quantity.
        """
        f = -state.qfrc_constraint[self.pusher_dofs]
        r = self._pusher_pos(state) - self._block_pose(state)[:2]
        tau = r[0] * f[1] - r[1] * f[0]
        return jnp.array([f[0], f[1], tau])

    def _ell_r(
        self, pose: jax.Array, pusher_pos: jax.Array, obj_ref: jax.Array
    ) -> jax.Array:
        """Robot stage cost ℓ_r: approach + push alignment (paper eq. 20-21)."""
        d_ee = jnp.sum((pusher_pos - pose[:2]) ** 2)
        approach = self.w_ee * jnp.clip(d_ee - self.r0**2, 0.0, None)

        to_object = pose[:2] - pusher_pos
        to_ref = obj_ref[:2] - pose[:2]
        cos_angle = jnp.sum(to_object * to_ref) / (
            jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
        )
        align = self.w_align * jnp.clip(self.gamma0 - cos_angle, 0.0, None)
        return approach + align

    def robot_running_cost(
        self, state: mjx.Data, control: jax.Array, obj_ref_t: jax.Array
    ) -> jax.Array:
        """Robot stage cost J_r = r_r||u||^2 + ℓ_o + ℓ_r + ℓ_c (paper eq. 17).

        The ADMM consensus penalty is *not* added here -- the ADMM layer adds
        it with the same `ConsensusSpace.penalty_cost` the object block uses.
        """
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        ell_o = se2_distance_sq(pose, self.goal, self.q_pos, self.q_theta)
        ell_r = self._ell_r(pose, pusher_pos, obj_ref_t)
        ell_c = se2_distance_sq(pose, obj_ref_t, self.q_pos, self.q_theta)
        return self.r_r * jnp.sum(control**2) + ell_o + ell_r + ell_c

    def robot_terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Heavier goal tracking, matching the object block's ℓ_f."""
        return se2_distance_sq(
            self._block_pose(state), self.goal, self.qf_pos, self.qf_theta
        )
