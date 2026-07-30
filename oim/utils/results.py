"""Persisting a run's hyperparameters, outcome, and state trajectories.

Two files per run, so that a run is reconstructable later without the video:

* `{stem}_metrics_{timestamp}.json` -- what the *algorithm* did: the
  hyperparameters it ran under, the per-step ADMM residuals and penalty, the
  goal errors, and whether it succeeded.
* `{stem}_states_{timestamp}.json` -- what the *world* did: the scene that
  never moves (goal, obstacles, object footprint) recorded once, and the
  things that do (object pose and twist, robot configuration and velocity,
  realized and agreed wrench) recorded every control step.

Both are produced from one `RunName`, so every artifact of a run -- these
two, the diagnostics plot, and the mp4 -- shares a single timestamp and
sorts together.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List, Optional

import numpy as np

from oim.objects import Box, Capsule, Circle, Polygon

# Per-step trajectory keys a run log may carry, in write order. Absent keys
# are skipped, so one writer serves both worlds: `qpos`/`qvel` exist only in
# MJX. The first six are defined identically in 2D and 3D, so the two
# worlds' files can be compared entry for entry.
_DYNAMIC_KEYS = (
    "time",
    "object_pose",  # [x, y, theta], world frame
    "object_velocity",  # [vx, vy, omega], world frame
    "robot_pos",  # world (x, y) of the robot's contact point
    "robot_vel",  # its world-frame velocity
    "robot_control",  # the command actually applied
    "wrench",  # A^r, the realized contact wrench
    "wrench_consensus",  # z_0, the agreed wrench for the executed step
    "qpos",  # full MuJoCo configuration (3D only)
    "qvel",  # full MuJoCo velocity (3D only)
    "object_plan",  # (H, 3) object block's predicted object trajectory
    "robot_plan",  # (H, 3) robot block's, same object -- only if requested
)


# Written into every states file. State series carry the initial condition
# and so run one longer than the input series -- the kind of off-by-one that
# silently misaligns an analysis months later, so the file states it.
_SCHEMA = {
    "indexing": (
        "state[i] --robot_control[i]--> state[i+1]. State series "
        "(time, object_pose, object_velocity, robot_pos, robot_vel, "
        "qpos, qvel) have steps_run+1 entries, entry 0 being the initial "
        "condition. Input and wrench series (robot_control, wrench, "
        "wrench_consensus) have steps_run entries, entry i being what was "
        "applied over the step from state[i] to state[i+1]."
    ),
    "frames": (
        "object_pose is [x, y, theta] in the world frame; "
        "object_velocity is its time derivative [vx, vy, omega]. "
        "object_footprint_body is in the object's body frame -- rotate by "
        "theta and translate by [x, y] for world coordinates. Wrenches are "
        "[f_x, f_y, tau] in the world frame about the object's origin."
    ),
    "plans": (
        "object_plan and robot_plan, present only when a run was asked for "
        "them, are the two ADMM blocks' predictions of the *same* object's "
        "trajectory over the horizon, made at that step: what the object "
        "block intends, and what the robot block's controls would produce. "
        "Shape (steps_run, H, 3), aligned with the input series -- entry i "
        "was computed from state[i]. Their divergence is the primal "
        "residual, resolved per timestep instead of summed into a scalar."
    ),
    "velocities": (
        "In 2D both velocities are difference quotients of the logged "
        "positions, which is exact there (the engine integrates pose with "
        "forward Euler), so entry 0 is zero. In 3D object_velocity is "
        "MuJoCo's own qvel for the block, and robot_vel is a difference "
        "quotient of the contact point, which has no qvel entry."
    ),
}


class RunName:
    """Names every artifact of one run off a single shared timestamp.

    Stamping each file as it is written would scatter one run's outputs
    across several timestamps whenever a step takes more than a second,
    which is normal here -- so the timestamp is taken once, at construction.
    """

    def __init__(self, *parts: str) -> None:
        """Build the stem from `parts` and fix the timestamp.

        Args:
            parts: Name components, joined with underscores, e.g.
                ("pusht3d", "xarm6", "admm").
        """
        self.stem = "_".join(parts)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def __call__(self, kind: Optional[str] = None) -> str:
        """`{stem}_{kind}_{timestamp}`, or `{stem}_{timestamp}` if no kind.

        Args:
            kind: What the file holds ("metrics", "states"). Omit for
                artifacts that are unambiguous by extension (plot, video).

        Returns:
            A filename stem, with no extension.
        """
        return "_".join(p for p in (self.stem, kind, self.timestamp) if p)


def _jsonable(value: Any) -> Any:
    """Convert JAX/NumPy arrays and scalars to plain JSON types."""
    if value is None:
        return None
    array = np.asarray(value)
    if array.ndim == 0:
        return array.item()
    return array.tolist()


def _shape_to_dict(shape: Any) -> Dict[str, Any]:
    """Serialize an `oim.objects` primitive to a self-describing dict.

    Keeps the constructor's own parameters rather than a sampled outline,
    so the shape can be rebuilt exactly rather than approximated.

    Args:
        shape: A `Circle`, `Box`, `Capsule`, or `Polygon`.

    Returns:
        A dict with a `type` tag and that primitive's parameters.

    Raises:
        TypeError: If the shape is not one of the four primitives.
    """
    if isinstance(shape, Circle):
        return {
            "type": "circle",
            "center": _jsonable(shape.center),
            "radius": float(shape.radius),
        }
    if isinstance(shape, Box):
        return {
            "type": "box",
            "center": _jsonable(shape.center),
            "half_extents": _jsonable(shape.half_extents),
            "angle": float(shape.angle),
        }
    if isinstance(shape, Capsule):
        return {
            "type": "capsule",
            "a": _jsonable(shape.a),
            "b": _jsonable(shape.b),
            "radius": float(shape.radius),
        }
    if isinstance(shape, Polygon):
        return {"type": "polygon", "vertices": _jsonable(shape.vertices)}
    raise TypeError(f"cannot serialize shape {type(shape).__name__}")


def save_run_metrics(
    output_dir: str,
    name: RunName,
    hyperparameters: Dict[str, Any],
    log: Dict[str, Any],
) -> str:
    """Save what the algorithm did: settings, residuals, goal errors.

    Works for both `oim.sim2d.run.run_2d`'s and `oim.sim3d.run.run_3d_admm`'s
    log dicts, since both share the same diagnostic keys.

    Args:
        output_dir: Directory to save into (created if missing).
        name: The run's `RunName`; the "metrics" file name comes from it.
        hyperparameters: Whatever was used to build the task/controller
            (horizon, rho, n_admm, num_samples, seed, ...).
        log: The run's log dict. Only per-step scalars and the outcome are
            kept here -- trajectories go to `save_run_states`.

    Returns:
        The path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name('metrics')}.json")
    results = {
        "hyperparameters": hyperparameters,
        "reached": bool(log["reached"]),
        "steps_run": len(log["primal_residual"]),
        "primal_residual": _jsonable(log["primal_residual"]),
        "dual_residual": _jsonable(log["dual_residual"]),
        "rho": _jsonable(log["rho"]),
        "pos_err": _jsonable(log["pos_err"]),
        "theta_err": _jsonable(log["theta_err"]),
    }
    with open(path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"saved metrics to {path}")
    return path


def save_run_states(
    output_dir: str,
    name: RunName,
    task: Any,
    log: Dict[str, Any],
    extra_static: Optional[Dict[str, Any]] = None,
) -> str:
    """Save what the world did: the fixed scene once, the motion per step.

    The split is deliberate. Obstacles, the goal and the object's footprint
    do not change during a run, so storing them once keeps the file small
    enough to hold every step of the things that do move -- which is what
    makes a run replayable from the log alone.

    Args:
        output_dir: Directory to save into (created if missing).
        name: The run's `RunName`; the "states" file name comes from it.
        task: A `ConsensusTask` exposing `goal`, `dt` and `object_model`
            (`PushT` or `PushT2D`). Read for the static scene only.
        log: The run's log dict. Every key of `_DYNAMIC_KEYS` present is
            written; the rest are ignored, so the two worlds' differing
            state representations both round-trip.
        extra_static: Additional fixed facts worth recording (embodiment,
            simulator timestep, scenario name, ...).

    Returns:
        The path written.
    """
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, f"{name('states')}.json")

    obj = task.object_model
    static: Dict[str, Any] = {
        "goal": _jsonable(task.goal),
        "control_dt": float(task.dt),
        "object_footprint_body": _jsonable(obj.footprint.vertices),
        "object_limit_surface_d": _jsonable(obj.D),
        "object_wrench_limit": _jsonable(obj.wrench_limit),
        "obstacles": [_shape_to_dict(s) for s in obj.obstacles.shapes],
    }
    if extra_static:
        static.update({k: _jsonable(v) for k, v in extra_static.items()})

    dynamic: Dict[str, List[Any]] = {
        key: _jsonable(log[key]) for key in _DYNAMIC_KEYS if key in log
    }
    payload = {
        "schema": _SCHEMA,
        "static": static,
        "dynamic": dynamic,
        "steps_run": len(log["primal_residual"]),
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"saved states to {path}")
    return path
