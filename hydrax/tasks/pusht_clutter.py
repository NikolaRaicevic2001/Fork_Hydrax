import jax
import jax.numpy as jnp
import mujoco
import numpy as np
from mujoco import mjx

from hydrax import ROOT
from hydrax.admm.consensus import WrenchConsensus
from hydrax.task_base import Task

GOAL = jnp.array([0.50, 0.48, jnp.pi / 4])
OBJECT_START = jnp.array([0.0, 0.0, 0.0])
ROBOT_START = jnp.array([-0.05, -0.06])

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


def goal_cost(
    pose: jnp.ndarray, goal: jnp.ndarray, w_pos: float, w_theta: float
) -> jnp.ndarray:
    """Quadratic SE(2) cost d^2(pose, goal), flattened per config convention."""
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


def obstacle_hinge_cost(
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
    """Capital-T outline in body frame, matching the MJCF crossbar + stem."""
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


class _ObjectModelStub:
    """Duck-typed stand-in for `mjx.Model`, exposing only what MPPI needs."""

    nu = 3


class ClutterObjectTask:
    """Object-level ADMM subproblem: closed-form quasi-static wrench dynamics.

    No MJX model -- `x_{t+1} = x_t + dt*D*w` (limit-surface compliance).
    Exposes `.model.nu`, `.dt`, `.u_min/u_max` so a real `hydrax.algs.mppi.MPPI`
    can be built directly against this task for sampling and weighting.
    """

    model = _ObjectModelStub()

    def __init__(self, consensus: WrenchConsensus, dt: float) -> None:
        """Set the limit-surface compliance, goal/obstacle weights, dt."""
        self.dt = dt
        self.consensus = consensus

        mu, mass, gravity = 0.4, 2.0, 9.81
        c, r_ls = 1.0, 0.06
        self.D = jnp.array(
            [
                1.0 / (mu * mass * gravity),
                1.0 / (mu * mass * gravity),
                1.0 / (c * r_ls * mu * mass * gravity),
            ]
        )
        # Wrench-sampling scale tied to mu*m*g (see HYDRAX_EXAMPLES_LOG.md).
        f_scale = 0.5 * mu * mass * gravity
        self.wrench_scale = jnp.array([f_scale, f_scale, f_scale * r_ls])
        self.u_min = -jnp.inf * jnp.ones(3)
        self.u_max = jnp.inf * jnp.ones(3)

        self.goal = GOAL
        self.q_pos, self.q_theta = 40.0, 10.0
        self.qf_pos, self.qf_theta = 500.0, 150.0
        self.r_o = 0.01
        self.w_obstacle, self.obstacle_margin = 60000.0, 0.015
        self.boundary_samples = _BOUNDARY_SAMPLES

    def step(self, pose: jnp.ndarray, wrench: jnp.ndarray) -> jnp.ndarray:
        """One Euler step of the quasi-static limit-surface dynamics."""
        new_pose = pose + self.dt * self.D * wrench
        return new_pose.at[2].set(_wrap_angle(new_pose[2]))

    def running_cost(self, pose: jnp.ndarray) -> jnp.ndarray:
        """Goal tracking + obstacle-margin hinge on the T-shape boundary."""
        world = pose[:2] + _rotate(pose[2], self.boundary_samples)
        cost = goal_cost(pose, self.goal, self.q_pos, self.q_theta)
        cost += obstacle_hinge_cost(
            world, self.w_obstacle, self.obstacle_margin
        )
        return cost

    def terminal_cost(self, pose: jnp.ndarray) -> jnp.ndarray:
        """Heavier goal tracking only, no obstacle term (ell_f in the paper)."""
        return goal_cost(pose, self.goal, self.qf_pos, self.qf_theta)


class ClutterRobotTask(Task):
    """Robot-level ADMM subproblem on the real clutter MJCF.

    Stage cost is the paper's `ell_o + ell_r + ell_c` (Sec05), minus the
    tilt term (2D planar pusher has no orientation DOF) and the 2D repo's
    ad-hoc obstacle hinge (real MJX contact already prevents penetration
    here) -- see HYDRAX_EXAMPLES_LOG.md for both deviations.
    """

    def __init__(
        self, consensus: WrenchConsensus, planning_dt: float, impl: str = "jax"
    ) -> None:
        """Load the clutter MJCF at the given planning timestep."""
        mj_model = mujoco.MjModel.from_xml_path(
            ROOT + "/models/pusht_clutter/scene.xml"
        )
        mj_model.opt.timestep = planning_dt
        super().__init__(mj_model, trace_sites=["pusher"], impl=impl)

        self.consensus = consensus
        pusher_x_dof = mj_model.joint("root_x").dofadr[0]
        pusher_y_dof = mj_model.joint("root_y").dofadr[0]
        self.pusher_dofs = jnp.array([pusher_x_dof, pusher_y_dof])

        self.goal = GOAL
        self.q_pos, self.q_theta = 40.0, 10.0
        self.qf_pos, self.qf_theta = 500.0, 150.0
        self.r_r = 0.05
        self.w_ee, self.r0 = 5.0, 0.05
        self.w_align, self.gamma0 = 2.0, jnp.cos(jnp.pi / 6)

    def _block_pose(self, state: mjx.Data) -> jax.Array:
        return state.qpos[:3]

    def _pusher_pos(self, state: mjx.Data) -> jax.Array:
        return state.qpos[3:5]

    def realized_wrench(self, state: mjx.Data) -> jax.Array:
        """A_r(U^r)_t: reaction wrench on the object from the pusher contact."""
        f = -state.qfrc_constraint[self.pusher_dofs]
        r = self._pusher_pos(state) - self._block_pose(state)[:2]
        tau = r[0] * f[1] - r[1] * f[0]
        return jnp.array([f[0], f[1], tau])

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

    def running_cost_admm(
        self,
        state: mjx.Data,
        control: jax.Array,
        z_t: jax.Array,
        gamma_r_t: jax.Array,
        obj_ref_t: jax.Array,
    ) -> jax.Array:
        """r_r*||u||^2 + ell_o + ell_r + ell_c + the ADMM penalty (Sec05)."""
        pose = self._block_pose(state)
        pusher_pos = self._pusher_pos(state)
        ell_o = goal_cost(pose, self.goal, self.q_pos, self.q_theta)
        ell_r = self._ell_r(pose, pusher_pos, obj_ref_t)
        ell_c = goal_cost(pose, obj_ref_t, self.q_pos, self.q_theta)
        penalty = self.consensus.penalty_cost(
            self.realized_wrench(state), z_t, gamma_r_t
        )
        return self.r_r * jnp.sum(control**2) + ell_o + ell_r + ell_c + penalty

    def terminal_cost_admm(self, state: mjx.Data) -> jax.Array:
        """Heavier goal tracking only, same ell_f as the object task."""
        return goal_cost(
            self._block_pose(state), self.goal, self.qf_pos, self.qf_theta
        )

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """Non-ADMM fallback: effort + object-goal tracking only."""
        pose = self._block_pose(state)
        return self.r_r * jnp.sum(control**2) + goal_cost(
            pose, self.goal, self.q_pos, self.q_theta
        )

    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Non-ADMM fallback terminal cost; same as `terminal_cost_admm`."""
        return self.terminal_cost_admm(state)
