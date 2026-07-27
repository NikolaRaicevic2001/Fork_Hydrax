import jax
import jax.numpy as jnp
import pytest
from mujoco import mjx

from hydrax.tasks.pusht import PushT


@pytest.mark.parametrize("impl", ["jax", "warp"])
@pytest.mark.parametrize("clutter", [False, True])
def test_task(impl: str, clutter: bool) -> None:
    """Set up the push T task.

    Args:
        impl: Which implementation to use ("jax" or "warp").
        clutter: Whether to load the cluttered scene.
    """
    task = PushT(impl=impl, clutter=clutter)

    state = task.make_data()
    assert isinstance(state, mjx.Data)
    state = state.replace(mocap_quat=jnp.array([[0.0, 1.0, 0.0, 0.0]]))
    state = jax.jit(mjx.forward)(task.model, state)

    pos = task._get_position_err(state)
    assert pos.shape == (3,)

    ori = task._get_orientation_err(state)
    assert ori.shape == (3,)

    ell = task.running_cost(state, jnp.zeros(2))
    assert ell.shape == ()

    phi = task.terminal_cost(state)
    assert phi.shape == ()


def test_clutter_consensus_task_methods() -> None:
    """Check shapes of the ConsensusTask (ADMM) methods, clutter=True only."""
    task = PushT(clutter=True, planning_dt=0.05)
    assert task.consensus_dim == 3

    state = task.make_data()
    state = jax.jit(mjx.forward)(task.model, state)

    obj_state = task.object_state_from_robot(state)
    assert obj_state.shape == (3,)

    w = jnp.array([1.0, 2.0, 0.1])
    next_state = task.object_dynamics(obj_state, w)
    assert next_state.shape == (3,)

    running = task.object_running_cost(obj_state, w)
    assert running.shape == ()

    terminal = task.object_terminal_cost(obj_state)
    assert terminal.shape == ()

    scale = task.object_action_scale()
    assert scale.shape == (3,)

    z_t = jnp.zeros(3)
    dual_t = jnp.zeros(3)
    rho = jnp.asarray(1.0)
    obj_ref_t = jnp.zeros(3)
    ell = task.robot_running_cost(
        state, jnp.zeros(2), z_t, dual_t, rho, obj_ref_t
    )
    assert ell.shape == ()

    phi = task.robot_terminal_cost(state)
    assert phi.shape == ()

    consensus_val = task.realized_consensus(state)
    assert consensus_val.shape == (3,)


if __name__ == "__main__":
    test_task("jax", False)
    test_task("jax", True)
