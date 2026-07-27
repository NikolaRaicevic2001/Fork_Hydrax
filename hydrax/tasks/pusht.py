from typing import Dict, Optional

import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.task_base import ConsensusTask, Task

# Clutter-variant goal pose and obstacle geometry (world-frame SE(2)), tuned
# against `models/pusht_clutter/scene.xml`.
GOAL = jnp.array([0.50, 0.48, jnp.pi / 4])

_OBS_CIRCLE_CENTER = jnp.array([0.08, 0.32])
_OBS_CIRCLE_RADIUS = 0.04
_OBS_BOX_CENTER = jnp.array([0.38, 0.10])
_OBS_BOX_HALF = jnp.array([0.04, 0.035])
_OBS_BOX_ANGLE = 0.25
_OBS_TRI_VERTICES = jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])


def _wrap_angle(a: jnp.ndarray) -> jnp.ndarray:
    return (a + jnp.pi) % (2.0 * jnp.pi) - jnp.pi


def _rotate(theta: jnp.ndarray, v: jnp.ndarray) -> jnp.ndarray:
    """Rotate 2D vector(s) v (..., 2) by angle(s) theta."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    vx, vy = v[..., 0], v[..., 1]
    return jnp.stack([c * vx - s * vy, s * vx + c * vy], axis=-1)


def _goal_cost(
    pose: jnp.ndarray, goal: jnp.ndarray, w_pos: float, w_theta: float
) -> jnp.ndarray:
    """Quadratic SE(2) cost d^2(pose, goal)."""
    diff_pos = pose[..., :2] - goal[:2]
    diff_theta = _wrap_angle(pose[..., 2] - goal[2])
    return w_pos * jnp.sum(diff_pos**2, axis=-1) + w_theta * diff_theta**2


def _circle_sdf(
    points: jnp.ndarray, center: jnp.ndarray, radius: float
) -> jnp.ndarray:
    return jnp.linalg.norm(points - center, axis=-1) - radius


def _box_sdf(
    points: jnp.ndarray,
    center: jnp.ndarray,
    half_extents: jnp.ndarray,
    angle: float,
) -> jnp.ndarray:
    local = _rotate(-angle, points - center)
    q = jnp.abs(local) - half_extents
    outside = jnp.linalg.norm(jnp.clip(q, 0.0, None), axis=-1)
    inside = jnp.clip(jnp.max(q, axis=-1), None, 0.0)
    return outside + inside


def _polygon_sdf(points: jnp.ndarray, vertices: jnp.ndarray) -> jnp.ndarray:
    """Signed distance to a closed polygon (negative inside), via winding."""
    n = vertices.shape[0]
    best_dist2 = jnp.full(points.shape[:-1], jnp.inf)
    winding = jnp.zeros(points.shape[:-1])
    for i in range(n):
        a, b = vertices[i], vertices[(i + 1) % n]
        ab = b - a
        t = jnp.clip(
            jnp.sum((points - a) * ab, axis=-1) / jnp.sum(ab * ab), 0.0, 1.0
        )
        proj = a + t[..., None] * ab
        dist2 = jnp.sum((points - proj) ** 2, axis=-1)
        best_dist2 = jnp.minimum(best_dist2, dist2)
        upward = (a[1] <= points[..., 1]) & (b[1] > points[..., 1])
        downward = (a[1] > points[..., 1]) & (b[1] <= points[..., 1])
        is_left = (b[0] - a[0]) * (points[..., 1] - a[1]) - (
            points[..., 0] - a[0]
        ) * (b[1] - a[1])
        winding += jnp.where(upward & (is_left > 0), 1.0, 0.0)
        winding += jnp.where(downward & (is_left < 0), -1.0, 0.0)
    sign = jnp.where(winding != 0, -1.0, 1.0)
    return sign * jnp.sqrt(best_dist2)


def _obstacle_hinge_cost(
    points: jnp.ndarray, weight: float, margin: float
) -> jnp.ndarray:
    """Squared-hinge penalty, summed over the 3 clutter obstacles and points."""
    cost = 0.0
    for d in (
        _circle_sdf(points, _OBS_CIRCLE_CENTER, _OBS_CIRCLE_RADIUS),
        _box_sdf(points, _OBS_BOX_CENTER, _OBS_BOX_HALF, _OBS_BOX_ANGLE),
        _polygon_sdf(points, _OBS_TRI_VERTICES),
    ):
        cost += weight * jnp.sum(jnp.clip(margin - d, 0.0, None) ** 2)
    return cost


def _t_shape_vertices_np() -> np.ndarray:
    """Capital-T outline in body frame, matching the clutter MJCF's block."""
    return np.array(
        [
            [-0.090, 0.045],
            [0.090, 0.045],
            [0.090, 0.015],
            [0.015, 0.015],
            [0.015, -0.105],
            [-0.015, -0.105],
            [-0.015, 0.015],
            [-0.090, 0.015],
        ]
    )


def _sample_boundary(vertices: np.ndarray, n_per_edge: int = 4) -> np.ndarray:
    n = len(vertices)
    pts = [
        vertices[i] + (vertices[(i + 1) % n] - vertices[i]) * k / n_per_edge
        for i in range(n)
        for k in range(n_per_edge)
    ]
    return np.asarray(pts)


_BOUNDARY_SAMPLES = jnp.asarray(_sample_boundary(_t_shape_vertices_np()))


class PushT(Task, ConsensusTask):
    """Push a T-shaped block to a desired pose, optionally through clutter.

    With `clutter=False` (default), loads the plain `models/pusht` scene and
    only supports ordinary sampling-based MPC (`running_cost`/`terminal_cost`).
    With `clutter=True`, loads `models/pusht_clutter` (static obstacles, a
    re-tuned model whose joint friction matches the closed-form limit-surface
    model below) and additionally implements the `ConsensusTask` methods
    needed for `hydrax.algs.admm.ADMM`.
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
            planning_dt: If given, overrides the model's simulation timestep
                (used to decouple the ADMM consensus planning rate from a
                finer real-execution timestep).
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

            # Goal-tracking weights (paper Sec. III-C).
            self.goal = GOAL
            self.q_pos, self.q_theta = 40.0, 10.0
            self.qf_pos, self.qf_theta = 500.0, 150.0

            # Robot-level cost weights (approach + alignment, paper eq. 20,
            # minus the tilt term -- this is a 2D planar pusher with no
            # end-effector orientation DOF).
            self.r_r = 0.05
            self.w_ee, self.r0 = 5.0, 0.05
            self.w_align, self.gamma0 = 2.0, jnp.cos(jnp.pi / 6)

            # Quasi-static limit-surface compliance (paper eq. 4-5). The
            # clutter MJCF's joint frictionloss values are deliberately
            # tuned to match these same mu*mass*gravity constants, so the
            # two ADMM blocks model consistent physics.
            mu, mass, gravity = 0.4, 2.0, 9.81
            c, r_ls = 1.0, 0.06
            self.D = jnp.array(
                [
                    1.0 / (mu * mass * gravity),
                    1.0 / (mu * mass * gravity),
                    1.0 / (c * r_ls * mu * mass * gravity),
                ]
            )
            f_scale = 0.5 * mu * mass * gravity
            self._object_action_scale = jnp.array(
                [f_scale, f_scale, f_scale * r_ls]
            )
            self.r_o = 0.01

            # Object-level obstacle avoidance (paper eq. 18, approximated
            # with a hand-rolled SDF hinge since the real MJX contact model
            # already prevents penetration).
            self.w_obstacle, self.obstacle_margin = 60000.0, 0.015
            self.boundary_samples = _BOUNDARY_SAMPLES

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
        position_err = self._get_position_err(state)
        orientation_err = self._get_orientation_err(state)
        close_to_block_err = self._close_to_block_err(state)

        position_cost = jnp.sum(jnp.square(position_err))
        orientation_cost = jnp.sum(jnp.square(orientation_err))
        close_to_block_cost = jnp.sum(jnp.square(close_to_block_err))

        return position_cost + orientation_cost + 0.01 * close_to_block_cost

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

    # -- ConsensusTask (ADMM) methods, only meaningful when clutter=True --

    def _block_pose(self, state: mjx.Data) -> jax.Array:
        return state.qpos[:3]

    def _pusher_pos(self, state: mjx.Data) -> jax.Array:
        return state.qpos[3:5]

    @property
    def consensus_dim(self) -> int:
        """Dimension of the consensus variable z_t = [f_x, f_y, tau]."""
        return 3

    def object_action_scale(self) -> jax.Array:
        """Per-dimension scale from wrench samples to physical units.

        Tied to mu*mass*gravity (see `__init__`).
        """
        return self._object_action_scale

    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """One Euler step of the quasi-static limit-surface dynamics."""
        new_state = obj_state + self.dt * self.D * w
        return new_state.at[2].set(_wrap_angle(new_state[2]))

    def object_running_cost(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """Goal tracking + obstacle-margin hinge + wrench effort."""
        world = obj_state[:2] + _rotate(obj_state[2], self.boundary_samples)
        cost = _goal_cost(obj_state, self.goal, self.q_pos, self.q_theta)
        cost += _obstacle_hinge_cost(
            world, self.w_obstacle, self.obstacle_margin
        )
        cost += self.r_o * jnp.sum(w**2)
        return cost

    def object_terminal_cost(self, obj_state: jax.Array) -> jax.Array:
        """Heavier goal tracking only, no obstacle term."""
        return _goal_cost(obj_state, self.goal, self.qf_pos, self.qf_theta)

    def _ell_r(
        self, pose: jax.Array, pusher_pos: jax.Array, obj_ref: jax.Array
    ) -> jax.Array:
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
        self,
        state: mjx.Data,
        control: jax.Array,
        z_t: jax.Array,
        dual_t: jax.Array,
        rho: jax.Array,
        obj_ref_t: jax.Array,
    ) -> jax.Array:
        """r_r*||u||^2 + ell_o + ell_r + ell_c + the ADMM penalty."""
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        ell_o = _goal_cost(pose, self.goal, self.q_pos, self.q_theta)
        ell_r = self._ell_r(pose, pusher_pos, obj_ref_t)
        ell_c = _goal_cost(pose, obj_ref_t, self.q_pos, self.q_theta)
        diff = self.realized_consensus(state) - z_t + dual_t
        penalty = 0.5 * rho * jnp.sum(diff**2)
        return self.r_r * jnp.sum(control**2) + ell_o + ell_r + ell_c + penalty

    def robot_terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Heavier goal tracking only, same as the object terminal cost."""
        return _goal_cost(
            self._block_pose(state), self.goal, self.qf_pos, self.qf_theta
        )

    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """A^r(U^r)_t: reaction wrench on the object from the pusher contact."""
        f = -state.qfrc_constraint[self.pusher_dofs]
        r = self._pusher_pos(state) - self._block_pose(state)[:2]
        tau = r[0] * f[1] - r[1] * f[0]
        return jnp.array([f[0], f[1], tau])

    def object_state_from_robot(self, state: mjx.Data) -> jax.Array:
        """Extract the object's SE(2) pose from the combined robot state."""
        return self._block_pose(state)
