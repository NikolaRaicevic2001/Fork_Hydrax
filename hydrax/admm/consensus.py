from abc import ABC, abstractmethod

import jax.numpy as jnp


class BaseConsensusSpace(ABC):
    """A consensus variable space for ADMM, shared between two subproblems."""

    @abstractmethod
    def penalty_cost(
        self, actual: jnp.ndarray, z: jnp.ndarray, dual: jnp.ndarray
    ) -> jnp.ndarray:
        """(rho/2) * ||actual - z + dual||^2, batched over a leading dim."""

    @abstractmethod
    def z_update(
        self,
        w_obj: jnp.ndarray,
        w_rob: jnp.ndarray,
        gamma_o: jnp.ndarray,
        gamma_r: jnp.ndarray,
    ) -> jnp.ndarray:
        """Consensus update z_t = 0.5*(w_obj + gamma_o + w_rob + gamma_r)."""

    @abstractmethod
    def dual_update(
        self, actual: jnp.ndarray, z: jnp.ndarray, dual: jnp.ndarray
    ) -> jnp.ndarray:
        """Dual step with anti-windup clamping."""


class WrenchConsensus(BaseConsensusSpace):
    """World-frame CoM wrench consensus z_t = [f_x, f_y, tau]; arrays (H, 3)."""

    dim = 3

    def __init__(self, horizon: int, rho: float, max_dual: float) -> None:
        """Set the horizon, penalty weight rho, and dual anti-windup clip."""
        self.horizon = horizon
        self.rho = rho
        self.max_dual = max_dual

    def penalty_cost(
        self, actual: jnp.ndarray, z: jnp.ndarray, dual: jnp.ndarray
    ) -> jnp.ndarray:
        """Collapses only the wrench dim; caller sums over horizon/samples."""
        diff = actual - z + dual
        return 0.5 * self.rho * jnp.sum(diff**2, axis=-1)

    def z_update(
        self,
        w_obj: jnp.ndarray,
        w_rob: jnp.ndarray,
        gamma_o: jnp.ndarray,
        gamma_r: jnp.ndarray,
    ) -> jnp.ndarray:
        """z_t = 0.5*(w_obj+gamma_o+w_rob+gamma_r), eq:consensus_update_icra."""
        return 0.5 * (w_obj + gamma_o + w_rob + gamma_r)

    def dual_update(
        self, actual: jnp.ndarray, z: jnp.ndarray, dual: jnp.ndarray
    ) -> jnp.ndarray:
        """Dual + (actual - z), clipped to [-max_dual, max_dual]."""
        return jnp.clip(dual + (actual - z), -self.max_dual, self.max_dual)
