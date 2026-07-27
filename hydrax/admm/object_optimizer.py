from typing import Any, Tuple

import jax
import jax.numpy as jnp

from hydrax.alg_base import SamplingBasedController, Trajectory
from hydrax.tasks.pusht_clutter import ClutterObjectTask


class ObjectOptimizer:
    """Object-level ADMM subproblem.

    Delegates sample/weight math to an injected sampling-based optimizer
    (e.g. `hydrax.algs.mppi.MPPI`) built against `ClutterObjectTask`. Rollout
    and cost use the task's closed-form dynamics directly, since there's no
    MJX model to route through.
    """

    def __init__(
        self, task: ClutterObjectTask, optimizer: SamplingBasedController
    ) -> None:
        """Pair the closed-form task with its injected sampling optimizer."""
        self.task = task
        self.optimizer = optimizer

    def _rollout(self, pose0: jax.Array, wrenches: jax.Array) -> jax.Array:
        """Scan the closed-form dynamics over a (H, 3) wrench sequence."""

        def step(pose, wrench):
            new_pose = self.task.step(pose, wrench)
            return new_pose, new_pose

        _, poses = jax.lax.scan(step, pose0, wrenches)
        return poses

    def optimize(
        self, pose0: jax.Array, params: Any, z: jax.Array, gamma_o: jax.Array
    ) -> Tuple[Any, jax.Array, jax.Array]:
        """One MPPI pass; returns updated params, W^o, and x_{0:H}^{o*}."""
        knots, params = self.optimizer.sample_knots(params)
        wrenches = knots * self.task.wrench_scale  # (K, H, 3)

        poses = jax.vmap(self._rollout, in_axes=(None, 0))(pose0, wrenches)

        running = jax.vmap(jax.vmap(self.task.running_cost))(poses[:, :-1])
        terminal = jax.vmap(self.task.terminal_cost)(poses[:, -1])
        effort = self.task.r_o * jnp.sum(wrenches**2, axis=-1)
        penalty = self.task.consensus.penalty_cost(wrenches, z, gamma_o)
        costs = (
            jnp.concatenate([running, terminal[:, None]], axis=1)
            + effort
            + penalty
        )

        rollouts = Trajectory(
            controls=knots,
            knots=knots,
            costs=costs,
            trace_sites=jnp.zeros((knots.shape[0], 1, 3)),
        )
        params = self.optimizer.update_params(params, rollouts)

        w_obj = params.mean * self.task.wrench_scale  # (H, 3)
        ref_poses = self._rollout(pose0, w_obj)
        return params, w_obj, ref_poses
