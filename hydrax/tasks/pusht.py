from typing import Dict, Literal, Optional

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

# xArm6 base placement (x, y, yaw about z), ground-mounted (z is not a free
# parameter -- the base sits directly on the floor, matching a real
# floor/table-mounted arm). Chosen by sweeping candidate offsets and joint
# configurations in models/xarm6_pusht_clutter/verify_reach.py and picking
# the one with the best worst-case reach across the block/goal/obstacle
# footprint (verified: every point in the workspace is reachable within a
# few cm, with the stick tip's own excluded self-collisions -- see
# models/xarm6/xarm6.xml -- otherwise unaffected by this placement).
XARM6_BASE_POS = (0.2, 0.75)
XARM6_BASE_YAW_DEG = -90.0


class PushT(Task, ConsensusTask):
    """Push a T-shaped block to a desired pose, optionally through clutter.

    With `clutter=False` (default), loads the plain `models/pusht` scene and
    supports ordinary sampling-based MPC (`running_cost`/`terminal_cost`).

    With `clutter=True`, loads `models/pusht_clutter` (static obstacles, and
    a model whose joint friction is tuned to match the analytic limit-surface
    object model) and additionally implements `ConsensusTask`, so it can be
    driven by `hydrax.algs.admm.ADMM`. The object-level subproblem is
    delegated to `hydrax.objects.PlanarPushingObject`.

    `robot` selects the embodiment used for the clutter scene: `"point"`
    (default) is the original free 2-DOF point-mass pusher
    (`models/pusht_clutter/pusht_clutter.xml`); `"xarm6"` swaps that for a
    real 6-DoF UFACTORY xArm6 with a rigid pushing-stick end-effector
    (`models/xarm6_pusht_clutter/`, `models/xarm6/xarm6.xml`), ground-mounted
    at `XARM6_BASE_POS`. Only meaningful with `clutter=True` -- there is no
    non-cluttered xArm6 scene. The two embodiments share every method below
    except the handful that read the "pusher position" or realize the
    contact wrench, which branch on `self.robot`; everything about the
    object side (goal, obstacles, limit-surface dynamics/costs) is exactly
    the same physics regardless of which robot is pushing.
    """

    def __init__(
        self,
        impl: str = "jax",
        clutter: bool = False,
        planning_dt: Optional[float] = None,
        robot: Literal["point", "xarm6"] = "point",
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            impl: The backend implementation for rollouts ("jax" or "warp").
            clutter: Whether to load the cluttered scene (with obstacles)
                and enable the ADMM `ConsensusTask` methods.
            planning_dt: If given, overrides the model's simulation timestep.
                Used to run the planner at a coarser rate than execution.
            robot: Which embodiment pushes the block in the clutter scene,
                `"point"` (default, the original free 2-DOF pusher) or
                `"xarm6"` (a real 6-DoF arm). Ignored (must be `"point"`)
                when `clutter=False`.
        """
        if robot not in ("point", "xarm6"):
            raise ValueError(f"robot must be 'point' or 'xarm6', got {robot!r}")
        if robot == "xarm6" and not clutter:
            raise ValueError("robot='xarm6' requires clutter=True")

        self.clutter = clutter
        self.robot = robot
        if not clutter:
            scene = "pusht/scene.xml"
        elif robot == "xarm6":
            scene = "xarm6_pusht_clutter/scene.xml"
        else:
            scene = "pusht_clutter/scene.xml"
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/" + scene)
        if planning_dt is not None:
            mj_model.opt.timestep = planning_dt

        if robot == "xarm6":
            # Ground-mounted base placement, not baked into xarm6.xml itself
            # (that file is a reusable, placement-agnostic robot asset) --
            # same pattern as overriding opt.timestep above: mutate the
            # loaded mj_model before it's handed to mjx.
            base_id = mj_model.body("xarm6_link_base").id
            mj_model.body_pos[base_id] = [*XARM6_BASE_POS, 0.0]
            yaw = jnp.deg2rad(XARM6_BASE_YAW_DEG)
            mj_model.body_quat[base_id] = [
                float(jnp.cos(yaw / 2)),
                0.0,
                0.0,
                float(jnp.sin(yaw / 2)),
            ]
            trace_site = "xarm6_tip"
        else:
            trace_site = "pusher"
        super().__init__(mj_model, trace_sites=[trace_site], impl=impl)

        # Sensor ids (defined identically in all three scenes).
        self.block_position_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "position"
        )
        self.block_orientation_sensor = mujoco.mj_name2id(
            mj_model, mujoco.mjtObj.mjOBJ_SENSOR, "orientation"
        )

        if clutter:
            if robot == "xarm6":
                # Block qpos addresses looked up explicitly, not assumed to
                # be qpos[:3] -- unlike pusht_clutter.xml (block declared
                # before the pusher), the composed xarm6 scene compiles the
                # arm's 5 joints first, so the block's SE(2) pose actually
                # lands at qpos[5:8].
                self.block_qpos_adr = jnp.array(
                    [
                        mj_model.joint("T_x").qposadr[0],
                        mj_model.joint("T_y").qposadr[0],
                        mj_model.joint("T_z").qposadr[0],
                    ]
                )
                self.tip_site_id = mj_model.site("xarm6_tip").id
                self.stick_body_id = mj_model.body("xarm6_stick").id
                self.block_body_id = mj_model.body("block").id
            else:
                pusher_x_dof = mj_model.joint("root_x").dofadr[0]
                pusher_y_dof = mj_model.joint("root_y").dofadr[0]
                self.pusher_dofs = jnp.array([pusher_x_dof, pusher_y_dof])

            # Analytic object-level subproblem. mu/mass are chosen so that
            # the friction-cone limit mu*m*g equals the block joints'
            # `frictionloss` in the MJCF -- the analytic model and the
            # simulated model then describe the same physics. Same for both
            # embodiments: this is physics of the block/table, not the
            # pusher.
            self.object_model = PlanarPushingObject(
                dt=self.dt,
                goal=GOAL,
                footprint=t_shape_footprint(),
                obstacles=CLUTTER_OBSTACLES,
                mu=0.4,
                mass=2.0,
                limit_surface_radius=0.06,
            )

            # Robot-level cost weights (paper eq. 20).
            self.r_r = 0.05
            self.w_ee, self.r0 = 20.0, 0.05
            self.w_align, self.gamma0 = 5.0, jnp.cos(jnp.pi / 6)
            # w_tilt/w_tip_z are not given numeric values in the paper --
            # untuned, same order of magnitude as w_align/w_ee since all are
            # soft shaping terms on the end-effector pose.
            self.w_tilt = 5.0
            self.w_tip_z = 50.0
            # Target tip height for good side contact: the block's own
            # (fixed) z -- it only translates in x/y and rotates about z,
            # so its z never changes during simulation, this is read once
            # here rather than hardcoded. Not `block thickness / 2` from a
            # z=0 floor as a first-principles calculation would give
            # (0.02 m, from the block geoms' z half-extent) -- this MJCF's
            # block actually rests slightly above that (z=0.03, a ~1cm
            # floor clearance already baked into pusht_clutter.xml, present
            # before any of this session's changes), so the block's own
            # position is the physically-correct target, not the idealized
            # formula.
            self.tip_target_z = float(mj_model.body("block").pos[2])
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
        """Position of the pusher relative to the block."""
        block_pos = self._block_pose(state)[:2]
        pusher_pos = self._pusher_pos(state)
        if self.robot == "point":
            pusher_pos = pusher_pos + jnp.array([0.0, 0.1])  # y bias
        return block_pos - pusher_pos

    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ) for plain (non-ADMM) MPC.

        For `robot="xarm6"`, this reuses the same approach/align/tilt
        shaping terms as the ADMM robot-level cost (`_ell_r`), with the
        fixed `self.goal` standing in for the object planner's reference
        (there is no separate object-level plan in plain MPC) -- otherwise
        the psi_tilt fix in `_ell_r` would have no visible effect here,
        since plain MPC never calls `robot_running_cost`. `robot="point"`
        keeps the original sensor-based formula untouched.
        """
        if self.robot == "xarm6":
            pose = self._block_pose(state)
            pusher_pos = self._pusher_pos(state)
            ell_o = se2_distance_sq(pose, self.goal, self.q_pos, self.q_theta)
            ell_r = self._ell_r(state, pose, pusher_pos, self.goal)
            return ell_o + ell_r
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
        if self.clutter and self.robot == "xarm6":
            # More headroom than the point-mass case: the arm has its own
            # (mostly-excluded) self-contact pairs in addition to the
            # stick/block/obstacle contacts, and a too-small allocation
            # silently drops contacts rather than erroring.
            return super().make_data(nconmax=256, naconmax=2048)
        if self.clutter:
            # Enough contact slots for the pusher, block, and 3 obstacles;
            # the default is too small and silently drops contacts.
            return super().make_data(nconmax=128, naconmax=1024)
        return super().make_data(nconmax=6000)

    # ------------------------------------------------------------------
    # ConsensusTask (ADMM) interface -- only meaningful when clutter=True
    # ------------------------------------------------------------------

    def _block_pose(self, state: mjx.Data) -> jax.Array:
        if self.robot == "xarm6":
            return state.qpos[self.block_qpos_adr]
        return state.qpos[:3]

    def _pusher_pos(self, state: mjx.Data) -> jax.Array:
        """World-frame (x, y) position of the pusher's contact point."""
        if self.robot == "xarm6":
            return state.site_xpos[self.tip_site_id, :2]
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

    def _stick_block_contact_force(self, state: mjx.Data) -> jax.Array:
        """World-frame linear contact force the stick imparts on the block,
        summed over every stick/block contact slot (there can be more than
        one for a capsule against a box).

        `mjx._src.support.contact_force` (the library function this would
        naturally call) cannot be `vmap`ped as-is: it branches on `condim`
        read from a per-contact array, which is fine for a single static
        `contact_id` but not under `vmap`. Every geom in this scene uses
        `condim=3` (the MJCF default, unchanged), so this reimplements just
        that one case -- the pyramidal-cone decode, condim==3 -- directly in
        a `vmap`-safe form. Verified directly: for a single, concrete
        contact index, this produces the exact same result as calling the
        library function itself; for a no-contact state, this returns
        exactly zero.

        Not yet fully cross-validated against `mj_contactForce`/MuJoCo-C
        the way the point-mass version below is: both engines agree on
        *where* the stick/block contact is (within ~2mm), but MuJoCo-C
        resolves it as a single contact point while MJX resolves the same
        capsule-vs-box pair as up to 4 candidate points -- the summed force
        here did not numerically match MuJoCo-C's single-point
        `mj_contactForce` value in testing. Plausibly a real discretization
        difference between the two engines' solves rather than a bug in
        this extraction (the mechanism itself is verified correct against
        MJX's own reference implementation), but unconfirmed -- worth a
        closer look before trusting this beyond a basic working ADMM loop.
        """
        c = state._impl.contact
        efc_force = state._impl.efc_force

        def _one(addr: jax.Array, mu: jax.Array, frame: jax.Array) -> jax.Array:
            pyramid = jax.lax.dynamic_slice(efc_force, (addr,), (4,))
            valid = (addr >= 0).astype(pyramid.dtype)
            normal = jnp.sum(pyramid) * valid
            t0 = (pyramid[0] - pyramid[1]) * mu[0] * valid
            t1 = (pyramid[2] - pyramid[3]) * mu[1] * valid
            return jnp.array([normal, t0, t1]) @ frame

        forces = jax.vmap(_one)(c.efc_address, c.friction, c.frame)
        body1 = jnp.asarray(self.model.geom_bodyid)[c.geom[:, 0]]
        body2 = jnp.asarray(self.model.geom_bodyid)[c.geom[:, 1]]
        mask = (
            (body1 == self.stick_body_id) & (body2 == self.block_body_id)
        ) | ((body1 == self.block_body_id) & (body2 == self.stick_body_id))
        return jnp.sum(jnp.where(mask[:, None], forces, 0.0), axis=0)

    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """A^r: the wrench the pusher applies to the object (paper eq. 23).

        For `robot="xarm6"`: linear force from `_stick_block_contact_force`
        (see its docstring for the verification caveat), moment arm from
        the tip position to the block's CoM (same convention as the
        `robot="point"` case below, for consistency -- not the literal
        contact point).

        For `robot="point"`: `qfrc_constraint` at the pusher's DOFs is the
        constraint force acting *on the pusher*; by Newton's third law its
        negation is the force the pusher applies to the object. Expressed in
        the world frame about the block's pose origin -- the same frame and
        reference point the object model integrates, and the same units, so
        both ADMM blocks report the identical physical quantity.
        """
        r = self._pusher_pos(state) - self._block_pose(state)[:2]
        if self.robot == "xarm6":
            f = self._stick_block_contact_force(state)
            tau = r[0] * f[1] - r[1] * f[0]
            return jnp.array([f[0], f[1], tau])
        f = -state.qfrc_constraint[self.pusher_dofs]
        tau = r[0] * f[1] - r[1] * f[0]
        return jnp.array([f[0], f[1], tau])

    def _tilt(self, state: mjx.Data) -> jax.Array:
        """psi_tilt(R_ee): end-effector tilt from vertical (paper eq. 22).

        Squared-sum of the roll/pitch angles extracted from the end-effector
        site's world-frame rotation matrix. Correctly zero for `robot=
        "point"`: its `pusher` site has no orientation DOF at all, so its
        rotation matrix is always identity (roll=pitch=0) -- no branch on
        `self.robot` is needed for that reason.
        """
        r_mat = state.site_xmat[self.trace_site_ids[0]]
        roll = jnp.arctan2(r_mat[2, 1], r_mat[2, 2])
        pitch = jnp.arctan2(
            -r_mat[2, 0], jnp.sqrt(r_mat[2, 1] ** 2 + r_mat[2, 2] ** 2)
        )
        return jnp.sqrt(roll**2 + pitch**2)

    def _tip_height_err(self, state: mjx.Data) -> jax.Array:
        """(z_tip - tip_target_z)^2: not in the paper -- an added term to
        keep the pusher near the block's own mid-height for good side
        contact, since nothing else in this cost constrains the tip's
        height at all (approach/align only ever look at xy). Correctly a
        no-op for `robot="point"`: its `pusher` site sits at the same fixed
        z as the block by construction, so this is identically zero -- no
        branch on `self.robot` needed.
        """
        z_tip = state.site_xpos[self.trace_site_ids[0], 2]
        return (z_tip - self.tip_target_z) ** 2

    def _ell_r(
        self,
        state: mjx.Data,
        pose: jax.Array,
        pusher_pos: jax.Array,
        obj_ref: jax.Array,
    ) -> jax.Array:
        """Robot stage cost ℓ_r: approach + align + tilt (paper eq. 20-22)
        + tip height (not in the paper, see `_tip_height_err`)."""
        d_ee = jnp.sum((pusher_pos - pose[:2]) ** 2)
        approach = self.w_ee * jnp.clip(d_ee - self.r0**2, 0.0, None)

        to_object = pose[:2] - pusher_pos
        to_ref = obj_ref[:2] - pose[:2]
        cos_angle = jnp.sum(to_object * to_ref) / (
            jnp.linalg.norm(to_object) * jnp.linalg.norm(to_ref) + 1e-6
        )
        align = self.w_align * jnp.clip(self.gamma0 - cos_angle, 0.0, None)

        tilt = self.w_tilt * self._tilt(state)
        tip_height = self.w_tip_z * self._tip_height_err(state)
        return approach + align + tilt + tip_height

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
        ell_r = self._ell_r(state, pose, pusher_pos, obj_ref_t)
        ell_c = se2_distance_sq(pose, obj_ref_t, self.q_pos, self.q_theta)
        return self.r_r * jnp.sum(control**2) + ell_o + ell_r + ell_c

    def robot_terminal_cost(self, state: mjx.Data) -> jax.Array:
        """Heavier goal tracking, matching the object block's ℓ_f."""
        return se2_distance_sq(
            self._block_pose(state), self.goal, self.qf_pos, self.qf_theta
        )
