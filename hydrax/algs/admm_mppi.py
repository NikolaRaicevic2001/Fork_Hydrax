from functools import partial
from typing import Any, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.alg_base import Trajectory
from hydrax.algs.mppi import MPPI
from hydrax.tasks.pusht_clutter import ClutterRobotTask


@dataclass
class ADMMTrajectory(Trajectory):
    """Trajectory with the realized wrench A_r(U^r)_t at each step (K, H, 3)."""

    wrenches: jax.Array


class ADMMMPPI(MPPI):
    """MPPI for the robot-level ADMM subproblem.

    Forks the parts of `SamplingBasedController`/`MPPI` that need the
    consensus variable, dual, and object reference threaded through as
    explicit jit arguments rather than task-owned state, which would either
    go stale across ADMM iterations or force a recompile every iteration.
    `alg_base.py`/`task_base.py` stay untouched.
    """

    def __init__(self, task: ClutterRobotTask, *args, **kwargs) -> None:
        """Build the MPPI controller for the robot-level ADMM subproblem."""
        super().__init__(task, *args, **kwargs)

    @partial(jax.vmap, in_axes=(None, None, None, 0, 0, None, None, None))
    def eval_rollouts_admm(
        self,
        model: mjx.Model,
        state: mjx.Data,
        controls: jax.Array,
        knots: jax.Array,
        z: jax.Array,
        gamma_r: jax.Array,
        obj_ref: jax.Array,
    ) -> Tuple[mjx.Data, ADMMTrajectory]:
        """Like `eval_rollouts`, but scores against the ADMM penalty.

        Also returns the realized wrench at each step.
        """

        def _scan_fn(x: mjx.Data, inputs):
            u, z_t, gamma_t, ref_t = inputs
            x = x.replace(ctrl=u)
            x = mjx.step(model, x)
            cost = self.dt * self.task.running_cost_admm(
                x, u, z_t, gamma_t, ref_t
            )
            wrench = self.task.realized_wrench(x)
            sites = self.task.get_trace_sites(x)
            return x, (x, cost, wrench, sites)

        final_state, (states, costs, wrenches, trace_sites) = jax.lax.scan(
            _scan_fn, state, (controls, z, gamma_r, obj_ref)
        )
        final_cost = self.task.terminal_cost_admm(final_state)
        final_trace_sites = self.task.get_trace_sites(final_state)

        costs = jnp.append(costs, final_cost)
        trace_sites = jnp.append(trace_sites, final_trace_sites[None], axis=0)

        return states, ADMMTrajectory(
            controls=controls,
            knots=knots,
            costs=costs,
            trace_sites=trace_sites,
            wrenches=wrenches,
        )

    def rollout_with_randomizations_admm(
        self,
        state: mjx.Data,
        tk: jax.Array,
        knots: jax.Array,
        rng: jax.Array,
        z: jax.Array,
        gamma_r: jax.Array,
        obj_ref: jax.Array,
    ) -> ADMMTrajectory:
        """Like `rollout_with_randomizations`, with z/gamma/obj_ref shared.

        They're the fixed target every sample is scored against, not
        something sampled, so they're shared across samples and
        randomizations rather than varying per rollout.
        """
        states = jax.vmap(lambda _, x: x, in_axes=(0, None))(
            jnp.arange(self.num_randomizations), state
        )
        if self.num_randomizations > 1:
            subrngs = jax.random.split(rng, self.num_randomizations)
            randomizations = jax.vmap(self.task.domain_randomize_data)(
                states, subrngs
            )
            states = states.tree_replace(randomizations)

        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, knots)

        _, rollouts = jax.vmap(
            self.eval_rollouts_admm,
            in_axes=(self.randomized_axes, 0, None, None, None, None, None),
        )(self.model, states, controls, knots, z, gamma_r, obj_ref)

        costs = self.risk_strategy.combine_costs(rollouts.costs)
        return rollouts.replace(
            costs=costs,
            controls=rollouts.controls[0],
            knots=rollouts.knots[0],
            trace_sites=rollouts.trace_sites[0],
            wrenches=rollouts.wrenches[0],
        )

    def optimize_with_consensus(
        self,
        state: mjx.Data,
        params: Any,
        z: jax.Array,
        gamma_r: jax.Array,
        obj_ref: jax.Array,
    ) -> Tuple[Any, ADMMTrajectory]:
        """One ADMM iteration's worth of `self.iterations` MPPI passes.

        Runs against a fixed z/gamma_r/obj_ref. The outer ADMM loop lives in
        `hydrax/admm/admm.py`, not here.
        """
        tk = params.tk

        def _scan_body(params: Any, _: Any):
            knots, params = self.sample_knots(params)
            knots = jnp.clip(knots, self.task.u_min, self.task.u_max)
            rng, dr_rng = jax.random.split(params.rng)
            rollouts = self.rollout_with_randomizations_admm(
                state, tk, knots, dr_rng, z, gamma_r, obj_ref
            )
            params = params.replace(rng=rng)
            params = self.update_params(params, rollouts)
            return params, rollouts

        params, rollouts = jax.lax.scan(
            f=_scan_body, init=params, xs=jnp.arange(self.iterations)
        )
        rollouts_final = jax.tree.map(lambda x: x[-1], rollouts)
        return params, rollouts_final

    def nominal_rollout_admm(
        self,
        state: mjx.Data,
        params: Any,
        z: jax.Array,
        gamma_r: jax.Array,
        obj_ref: jax.Array,
    ) -> jax.Array:
        """Re-simulate `params.mean` alone.

        MPPI's softmax update blends all K samples into a new mean, so
        there's no single winning rollout already computed to read
        A_r(U^r)_t off of. Uses `self.task.model`, not `self.model` (a
        num_randomizations-way ensemble when num_randomizations > 1, which
        doesn't match a single state here).
        """
        tk = params.tk
        tq = jnp.linspace(tk[0], tk[-1], self.ctrl_steps)
        controls = self.interp_func(tq, tk, params.mean[None, ...])[0]

        def _scan_fn(x: mjx.Data, inputs):
            u, z_t, gamma_t, ref_t = inputs
            del z_t, gamma_t, ref_t
            x = x.replace(ctrl=u)
            x = mjx.step(self.task.model, x)
            wrench = self.task.realized_wrench(x)
            return x, wrench

        _, wrenches = jax.lax.scan(
            _scan_fn, state, (controls, z, gamma_r, obj_ref)
        )
        return wrenches
