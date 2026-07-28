from abc import ABC, abstractmethod
from typing import Dict, Sequence

import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx


class Task(ABC):
    """An abstract task interface, defining the dynamics and cost functions.

    The task is a discrete-time optimal control problem of the form

        minᵤ ϕ(x_{T+1}) + ∑ₜ ℓ(xₜ, uₜ)
        s.t. xₜ₊₁ = f(xₜ, uₜ)

    where the dynamics f(xₜ, uₜ) are defined by a MuJoCo model, and the costs
    ℓ(xₜ, uₜ) and ϕ(x_{T+1}) are defined by the task instance itself.
    """

    def __init__(
        self,
        mj_model: mujoco.MjModel,
        trace_sites: Sequence[str] | None = None,
        impl: str = "jax",
    ) -> None:
        """Set the model and simulation parameters.

        Args:
            mj_model: The MuJoCo model to use for simulation.
            trace_sites: A list of site names to visualize with traces.
            impl: The backend implementation for rollouts ("jax" for standard
                  MJX or "warp" for MjWarp).

        Note: many other simulator parameters, e.g., simulator time step,
              Newton iterations, etc., are set in the model itself.
        """
        assert isinstance(mj_model, mujoco.MjModel)
        self.mj_model = mj_model
        self.model = mjx.put_model(mj_model, impl=impl)

        # Set actuator limits
        self.u_min = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 0],
            -jnp.inf,
        )
        self.u_max = jnp.where(
            mj_model.actuator_ctrllimited,
            mj_model.actuator_ctrlrange[:, 1],
            jnp.inf,
        )

        # Simulation timestep
        self.dt = mj_model.opt.timestep

        # Get site IDs for points we want to trace
        trace_sites = trace_sites or []
        self.trace_site_ids = jnp.array(
            [mj_model.site(name).id for name in trace_sites]
        )

    @abstractmethod
    def running_cost(self, state: mjx.Data, control: jax.Array) -> jax.Array:
        """The running cost ℓ(xₜ, uₜ).

        Args:
            state: The current state xₜ.
            control: The control action uₜ.

        Returns:
            The scalar running cost ℓ(xₜ, uₜ)
        """
        pass

    @abstractmethod
    def terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The terminal cost ϕ(x_T).

        Args:
            state: The final state x_T.

        Returns:
            The scalar terminal cost ϕ(x_T).
        """
        pass

    def get_trace_sites(self, state: mjx.Data) -> jax.Array:
        """Get the positions of the trace sites at the current time step.

        Args:
            state: The current state xₜ.

        Returns:
            The positions of the trace sites at the current time step.
        """
        if len(self.trace_site_ids) == 0:
            return jnp.zeros((0, 3))

        return state.site_xpos[self.trace_site_ids]

    def domain_randomize_model(self, rng: jax.Array) -> Dict[str, jax.Array]:
        """Generate randomized model parameters for domain randomization.

        Returns a dictionary of randomized model parameters, that can be used
        with `mjx.Model.tree_replace` to create a new randomized model.

        For example, we might set the `model.geom_friction` values by returning
        `{"geom_friction": new_frictions, ...}`.

        The default behavior is to return an empty dictionary, which means no
        randomization is applied.

        Args:
            rng: A random number generator key.

        Returns:
            A dictionary of randomized model parameters.
        """
        return {}

    def domain_randomize_data(
        self, data: mjx.Data, rng: jax.Array
    ) -> Dict[str, jax.Array]:
        """Generate randomized data elements for domain randomization.

        This is the place where we could randomize the initial state and other
        `data` elements. Like `domain_randomize_model`, this method should
        return a dictionary that can be used with `mjx.Data.tree_replace`.

        Args:
            data: The base data instance holding the current state.
            rng: A random number generator key.

        Returns:
            A dictionary of randomized data elements.
        """
        return {}

    def make_data(self, **kwargs) -> mjx.Data:
        """Create a new state consistent with this task.

        By default, this just creates a new `mjx.Data` instance from the model.
        Specific tasks can override this method to set parameters that must be
        adjusted per task, e.g., nconmax and naconmax.

        TODO(vincekurtz): figure out a smarter place to set naconmax and njmax.
        N.B. when performing parallel rollouts with MjWarp, naconmax and
        njmax need to be set high enough to support constraint solving across
        *all* rollouts. This means that these parameters scale with the number
        of parallel rollouts/samples, as well as the complexity of the task.

        Args:
            **kwargs: Additional keyword arguments to pass to `mjx.make_data`.

        Returns:
            A new `mjx.Data` instance for this task.
        """
        return mjx.make_data(self.mj_model, impl=self.model.impl, **kwargs)


class ConsensusTask(ABC):
    """Mixin for tasks that support ADMM object/robot consensus.

    A task combines this with `Task` (e.g. `class MyTask(Task, ConsensusTask)`)
    to expose the object-level closed-form subproblem, the robot-level
    ADMM-penalized cost, and the consensus variable that ties them together,
    so that `oim.algs.admm.ADMM` can coordinate the two. Tasks that don't
    need ADMM are unaffected by this mixin's existence.

    The consensus variable z_t is generic: it might be a contact wrench, an
    object pose, or something else entirely. `consensus_dim` declares its
    dimension, and `realized_consensus` extracts it from the robot's rollout
    state.
    """

    @property
    @abstractmethod
    def consensus_dim(self) -> int:
        """The dimension of the consensus variable z_t."""

    @abstractmethod
    def object_dynamics(self, obj_state: jax.Array, w: jax.Array) -> jax.Array:
        """Closed-form object-side dynamics x^o_{t+1} = f^o(x^o_t, w_t).

        Args:
            obj_state: The object's configuration x^o_t.
            w: The object-level consensus decision w^o_t.

        Returns:
            The next object configuration x^o_{t+1}.
        """

    @abstractmethod
    def object_running_cost(
        self, obj_state: jax.Array, w: jax.Array
    ) -> jax.Array:
        """The object-level running cost ℓ_o(x^o_t).

        Includes any effort regularization on the consensus decision w^o_t.
        """

    @abstractmethod
    def object_terminal_cost(self, obj_state: jax.Array) -> jax.Array:
        """The object-level terminal cost ℓ_f(x^o_H)."""

    def object_action_scale(self) -> jax.Array:
        """Per-dimension scale for the object optimizer's sampled knots.

        Applied before they're treated as the physical consensus decision
        w^o_t. Defaults to no scaling; override when the sampler's natural
        units differ substantially across dimensions (e.g. force vs torque).
        """
        return jnp.ones(self.consensus_dim)

    def consensus_scale(self) -> jax.Array:
        """Characteristic per-dimension magnitude of the consensus variable.

        Used to normalize the ADMM penalty and residuals so that `rho` and
        the residual tolerances are scale-free and the penalty is comparable
        in magnitude to the task costs. Defaults to ones (no normalization);
        override with e.g. the largest physically transmissible wrench.
        """
        return jnp.ones(self.consensus_dim)

    @abstractmethod
    def robot_running_cost(
        self, state: mjx.Data, control: jax.Array, obj_ref_t: jax.Array
    ) -> jax.Array:
        """The robot-level running cost J_r (paper eq. 17).

        This is the task's own cost only — ℓ_o + ℓ_r + ℓ_c plus any control
        effort term. Do **not** add the ADMM consensus penalty here: the
        ADMM layer adds it via `ConsensusSpace.penalty_cost`, the same
        function the object block uses, so both blocks are guaranteed to
        score the consensus variable identically.

        Args:
            state: The robot's current MJX state x^r_t.
            control: The control action u^r_t.
            obj_ref_t: The object planner's current reference x^{o*}_t.

        Returns:
            The scalar robot-level running cost.
        """

    @abstractmethod
    def robot_terminal_cost(self, state: mjx.Data) -> jax.Array:
        """The robot-level terminal cost (shared with the object goal)."""

    @abstractmethod
    def realized_consensus(self, state: mjx.Data) -> jax.Array:
        """The extraction map A^r: robot state -> realized consensus value."""

    @abstractmethod
    def object_state_from_robot(self, state: mjx.Data) -> jax.Array:
        """Extract the object's own configuration from the robot state.

        Reads x^o_t out of the robot's combined MJX state, to seed the
        object subproblem each real step.
        """
