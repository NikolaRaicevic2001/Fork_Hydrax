from typing import Dict, Literal, Optional, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx

from oim import ROOT
from oim.objects import PlanarPushingObject, se2_distance_sq
from oim.task_base import ConsensusTask, Task
from oim.utils.scenes import SCENES


class PushT(Task, ConsensusTask):
    """Push a T-shaped block to a desired pose, optionally through clutter.

    With `clutter=False` (default), loads the plain `models/pusht` scene and
    supports ordinary sampling-based MPC (`running_cost`/`terminal_cost`).

    With `clutter=True`, loads `models/pusht_clutter` (static obstacles, and
    a model whose joint friction is tuned to match the analytic limit-surface
    object model) and additionally implements `ConsensusTask`, so it can be
    driven by `oim.algs.admm.ADMM`. The object-level subproblem is
    delegated to `oim.objects.PlanarPushingObject`.

    `robot` selects the embodiment: `"point"` (default) is the original
    free 2-DOF point-mass pusher; `"xarm6"` swaps that for a real 6-DoF
    UFACTORY xArm6 with a rigid pushing-stick end-effector. Only meaningful
    with `clutter=True` -- there is no non-cluttered xArm6 scene. The two
    embodiments share every method below except the handful that read the
    "pusher position" or realize the contact wrench, which branch on
    `self.robot`; everything about the object side (goal, obstacles,
    limit-surface dynamics/costs) is exactly the same physics regardless of
    which robot is pushing.

    `env` selects which scene to load, by name, from the
    `oim.utils.scenes.SCENES` registry -- `PushT` itself holds no
    scene-specific data or branching; it asks the registry for one
    `SceneSpec` (MJCF path per embodiment, goal, obstacles, footprint, and
    the xArm6 base placement) and wraps cost functions/ADMM plumbing around
    whatever it's handed. Adding an environment is a new `SCENES` entry
    plus its own MJCF, never a change here.
    """

    def __init__(
        self,
        impl: str = "jax",
        clutter: bool = False,
        planning_dt: Optional[float] = None,
        robot: Literal["point", "xarm6"] = "point",
        consensus_source: Literal["twist", "contact"] = "twist",
        env: str = "clutter",
        goal: Optional[Sequence[float]] = None,
    ) -> None:
        """Load the MuJoCo model and set task parameters.

        Args:
            impl: The backend implementation for rollouts ("jax" or "warp").
            clutter: Whether to load `env`'s scene (with obstacles) and
                enable the ADMM `ConsensusTask` methods.
            planning_dt: If given, overrides the model's simulation timestep.
                Used to run the planner at a coarser rate than execution.
            robot: Which embodiment pushes the block, `"point"` (default,
                the original free 2-DOF pusher) or `"xarm6"` (a real 6-DoF
                arm). Ignored (must be `"point"`) when `clutter=False`.
            consensus_source: How the robot block estimates A^r. `"twist"`
                (default) inverts the limit-surface relation, `w = D^-1
                xdot^o`; works on both backends and both embodiments, and is
                continuous through contact breaks. `"contact"` reads the
                simulator's constraint force literally, matching the paper's
                wording, but is only valid for `robot="point"`.
            env: Which scene to load, by name from
                `oim.utils.scenes.SCENES` (only meaningful with
                `clutter=True`). Must support `robot`, or raises.
            goal: Overrides the scene's own goal pose, world-frame SE(2)
                `[x, y, theta]`. Used by `examples/poses/<task>.yaml` to
                run one scene against several goals. The goal marker in
                the MJCF is a mocap body, moved separately by
                `oim.sim3d.build`; this sets what the *costs* aim at.
        """
        if robot not in ("point", "xarm6"):
            raise ValueError(f"robot must be 'point' or 'xarm6', got {robot!r}")
        if robot == "xarm6" and not clutter:
            raise ValueError("robot='xarm6' requires clutter=True")
        if consensus_source not in ("twist", "contact"):
            raise ValueError(
                "consensus_source must be 'twist' or 'contact', got "
                f"{consensus_source!r}"
            )
        if consensus_source == "contact" and robot != "point":
            raise ValueError(
                "consensus_source='contact' is only valid for robot='point'; "
                "an articulated arm's contact force appears as J^T f spread "
                "across its joints, not at a single pair of DOFs."
            )

        self.clutter = clutter
        self.robot = robot
        self.consensus_source = consensus_source
        self.env = env
        if not clutter:
            scene_path = "pusht/scene.xml"
        else:
            if env not in SCENES:
                raise ValueError(
                    f"env={env!r} is not in oim.utils.scenes.SCENES "
                    f"(available: {sorted(SCENES)})"
                )
            spec = SCENES[env]
            scene_path = spec.mjcf_scene(robot)
        mj_model = mujoco.MjModel.from_xml_path(ROOT + "/models/" + scene_path)
        if planning_dt is not None:
            mj_model.opt.timestep = planning_dt

        if robot == "xarm6":
            # Ground-mounted base placement, not baked into xarm6.xml itself
            # (that file is a reusable, placement-agnostic robot asset) --
            # same pattern as overriding opt.timestep above: mutate the
            # loaded mj_model before it's handed to mjx. Each scene has its
            # own mount, since the workspace moves between them.
            base_id = mj_model.body("xarm6_link_base").id
            mj_model.body_pos[base_id] = [*spec.xarm6_base_pos, 0.0]
            yaw = jnp.deg2rad(spec.xarm6_base_yaw_deg)
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

            # The block's own velocity DOFs, used by the default
            # ("twist") consensus extraction. Looked up by joint name so
            # it is correct for both embodiments' qpos/qvel layouts.
            self.block_dofs = jnp.array(
                [
                    mj_model.joint("T_x").dofadr[0],
                    mj_model.joint("T_y").dofadr[0],
                    mj_model.joint("T_z").dofadr[0],
                ]
            )

            # goal/obstacles/footprint come from the scene registry (see
            # oim.utils.scenes) -- the only things that differ between
            # scenes. mu/mass/limit_surface_radius stay fixed (physics of
            # the block/table, not its shape), chosen so the friction-cone
            # limit mu*m*g equals the block joints' `frictionloss` in the
            # MJCF for every scene's geoms.
            # One goal pose feeds both blocks' costs; a pose file overrides
            # it per run. Both must read the same array or the two ADMM
            # blocks would negotiate a wrench toward different targets.
            goal_pose = (
                spec.goal if goal is None else jnp.asarray(goal, dtype=float)
            )
            self.object_model = PlanarPushingObject(
                dt=self.dt,
                goal=goal_pose,
                footprint=spec.footprint(),
                obstacles=spec.obstacles,
                mu=0.4,
                mass=2.0,
                limit_surface_radius=0.06,
            )

            # Robot-level cost weights (paper eq. 20).
            self.r_r = 0.05
            self.w_ee, self.r0 = 40.0, 0.02
            self.w_align, self.gamma0 = 15.0, jnp.cos(jnp.pi / 12)
            # w_tilt/w_tip_z: not in the paper, untuned, same order of
            # magnitude as w_align/w_ee. w_tilt raised from 5.0 now that
            # _tilt's sign bug is fixed (task 11/12) -- at 5.0 the tip still
            # averaged ~35 degrees off vertical. Raised again from 20.0 to
            # 50.0 (still visibly going near-horizontal under ADMM), then
            # brought down to 30.0 -- 50.0 cost too much task performance.
            self.w_tilt = 30.0
            self.w_tip_z = 8.0
            # Target tip height: the block's own resting z, read from the
            # model rather than hardcoded.
            self.tip_target_z = float(mj_model.body("block").pos[2])
            self.q_pos, self.q_theta = 40.0, 10.0
            self.qf_pos, self.qf_theta = 500.0, 150.0
            self.goal = goal_pose

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

        `robot="xarm6"` reuses `_ell_r`'s approach/align/tilt shaping with
        `self.goal` standing in for the object planner's reference (plain
        MPC has no object-level plan). `robot="point"` uses the original
        sensor-based formula.
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

    @property
    def block_qpos_indices(self) -> jax.Array:
        """Where the object's SE(2) pose sits in `qpos`, per embodiment.

        The arm's five joints compile first, so the block lands at
        `qpos[5:8]`; the point pusher's scene declares the block first, so
        it is `qpos[:3]`. Read by anything writing a start pose in.
        """
        if self.robot == "xarm6":
            return self.block_qpos_adr
        return jnp.array([0, 1, 2])

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

    def _consensus_from_twist(self, state: mjx.Data) -> jax.Array:
        """A^r via the limit-surface relation `xdot^o = D w^o` (paper eq. 4).

        Inverted to recover the wrench that produced the observed twist.
        Default estimator: backend-agnostic (needs only `qvel`), robot-
        agnostic (no contact enumeration), and continuous (contact forces
        are exactly zero between contacts, so `_consensus_from_contact`
        gives a chattery signal; this doesn't).
        """
        return self.object_model.wrench_limit * state.qvel[self.block_dofs]

    def _consensus_from_contact(self, state: mjx.Data) -> jax.Array:
        """A^r read literally from the simulator's constraint force.

        `qfrc_constraint` at the pusher's DOFs is the force acting on the
        pusher; its negation is the force applied to the object (Newton's
        third law). Point pusher only: relies on the pusher's DOFs being
        exactly the two translational DOFs in contact with the block, which
        doesn't hold for an articulated arm.
        """
        f = -state.qfrc_constraint[self.pusher_dofs]
        r = self._pusher_pos(state) - self._block_pose(state)[:2]
        tau = r[0] * f[1] - r[1] * f[0]
        return jnp.array([f[0], f[1], tau])

    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """A^r: the wrench the robot imparts on the object (paper eq. 23).

        Expressed in the world frame about the block's pose origin, in N and
        N·m -- the same frame, reference point and units the object block's
        A^o uses, so both ADMM blocks report the identical physical quantity.

        Which estimator is used is set by `consensus_source` on the task; see
        `_consensus_from_twist` (default) and `_consensus_from_contact`.

        Clipped to `consensus_scale()`: a rigid-body contact solver can
        report a one-step force or an implied velocity far past the
        friction-cone limit at contact onset (measured up to ~16x on this
        task), which no sustained push can exceed. Left unclipped, that
        outlier drags the consensus average z outside the object block's own
        feasible bound -- which it can never match, since its actions are
        already confined to that bound -- and the resulting disagreement
        persists for several steps after the spike itself is gone (task 10).
        """
        raw = (
            self._consensus_from_contact(state)
            if self.consensus_source == "contact"
            else self._consensus_from_twist(state)
        )
        scale = self.consensus_scale()
        return jnp.clip(raw, -scale, scale)

    def _tilt(self, state: mjx.Data) -> jax.Array:
        """psi_tilt(R_ee): end-effector tilt from vertical (paper eq. 22).

        Angle between the tip site's z-axis and world -z (straight down --
        the pushing stick's intended pointing direction): 0 when vertical
        and correctly oriented, pi when upside down. Identically zero for
        `robot="point"` (no orientation DOF).

        A previous roll/pitch-based formula measured deviation from the
        site's z-axis pointing *up*, so a correctly vertical, downward-
        pointing stick scored ~180 degrees "tilt" instead of ~0 -- verified
        directly against the reach-swept starting pose, whose site rotation
        has z-axis world-frame component [0.41, -0.11, -0.90] (pointing
        down) yet scored ~182 degrees under the old formula. `w_tilt` was
        therefore driving the tip *away* from vertical, not toward it
        (task 11/12).
        """
        r_mat = state.site_xmat[self.trace_site_ids[0]]
        return jnp.arccos(jnp.clip(-r_mat[2, 2], -1.0, 1.0))

    def _tip_height_err(self, state: mjx.Data) -> jax.Array:
        """(z_tip - tip_target_z)^2: not in the paper.

        Keeps the pusher at the block's height for side contact.
        Identically zero for `robot="point"`.
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
        """Robot stage cost ℓ_r (paper eq. 20-22).

        approach + align + tilt + tip height (the last not in the paper,
        see `_tip_height_err`).
        """
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
