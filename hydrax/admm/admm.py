from typing import Any, Dict, Tuple

import jax
import jax.numpy as jnp
from flax.struct import dataclass
from mujoco import mjx

from hydrax.admm.consensus import WrenchConsensus
from hydrax.admm.object_optimizer import ObjectOptimizer
from hydrax.algs.admm_mppi import ADMMMPPI


@dataclass
class ADMMState:
    """Warm-started state carried between real MPC steps.

    object_params/robot_params are each the injected optimizer's own params
    type (e.g. MPPIParams); z/gamma_o/gamma_r are (H, 3) arrays.
    """

    object_params: Any
    robot_params: Any
    z: jax.Array
    gamma_o: jax.Array
    gamma_r: jax.Array


def _shift(seq: jax.Array) -> jax.Array:
    """Receding-horizon shift: seq[t] <- seq[t+1], last slot <- 0."""
    return jnp.concatenate([seq[1:], jnp.zeros_like(seq[:1])], axis=0)


class ADMMController:
    """Coordinates the object- and robot-level MPPI subproblems.

    Consensus is reached on the contact wrench (Algorithm 4, root_icra.tex
    Sec05); the outer ADMM loop runs as a plain Python loop over `n_admm`
    iterations, jitting only the object/robot updates and consensus math.
    """

    def __init__(
        self,
        object_ctrl: ObjectOptimizer,
        robot_ctrl: ADMMMPPI,
        consensus: WrenchConsensus,
        n_admm: int,
        eps_r: float,
        eps_s: float,
    ) -> None:
        """Store the two subproblem controllers and jit their update steps."""
        self.object_ctrl = object_ctrl
        self.robot_ctrl = robot_ctrl
        self.consensus = consensus
        self.n_admm = n_admm
        self.eps_r = eps_r
        self.eps_s = eps_s

        self._object_optimize = jax.jit(object_ctrl.optimize)
        self._robot_optimize = jax.jit(robot_ctrl.optimize_with_consensus)
        self._robot_nominal_rollout = jax.jit(robot_ctrl.nominal_rollout_admm)
        self._z_update = jax.jit(consensus.z_update)
        self._dual_update = jax.jit(consensus.dual_update)

    def control_step(
        self,
        object_pose0: jax.Array,
        robot_state: mjx.Data,
        admm_state: ADMMState,
    ) -> Tuple[ADMMState, jax.Array, Dict[str, Any]]:
        """Run up to `n_admm` object/robot/consensus updates for one step."""
        object_params = admm_state.object_params.replace(
            mean=_shift(admm_state.object_params.mean)
        )
        robot_params = admm_state.robot_params.replace(
            mean=_shift(admm_state.robot_params.mean)
        )
        z = _shift(admm_state.z)
        gamma_o = _shift(admm_state.gamma_o)
        gamma_r = _shift(admm_state.gamma_r)

        obj_ref = jnp.tile(object_pose0, (z.shape[0], 1))
        w_obj = jnp.zeros_like(z)
        w_rob = jnp.zeros_like(z)
        n_ran = 0
        primal_res = jnp.inf
        dual_res = jnp.inf

        for _ in range(self.n_admm):
            object_params, w_obj, obj_ref = self._object_optimize(
                object_pose0, object_params, z, gamma_o
            )
            robot_params, _ = self._robot_optimize(
                robot_state, robot_params, z, gamma_r, obj_ref
            )
            w_rob = self._robot_nominal_rollout(
                robot_state, robot_params, z, gamma_r, obj_ref
            )

            z_new = self._z_update(w_obj, w_rob, gamma_o, gamma_r)
            gamma_o = self._dual_update(w_obj, z_new, gamma_o)
            gamma_r = self._dual_update(w_rob, z_new, gamma_r)

            primal_res = float(
                jnp.linalg.norm(jnp.concatenate([w_obj - z_new, w_rob - z_new]))
            )
            dual_res = float(jnp.linalg.norm(self.consensus.rho * (z_new - z)))
            z = z_new
            n_ran += 1
            if primal_res <= self.eps_r and dual_res <= self.eps_s:
                break

        new_state = ADMMState(
            object_params=object_params,
            robot_params=robot_params,
            z=z,
            gamma_o=gamma_o,
            gamma_r=gamma_r,
        )
        u0 = robot_params.mean[0]
        info = {
            "admm_iters": n_ran,
            "primal_residual": primal_res,
            "dual_residual": dual_res,
            "obj_ref": obj_ref,
            "w_obj": w_obj,
            "w_rob": w_rob,
        }
        return new_state, u0, info
