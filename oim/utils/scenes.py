"""Registry of the 3D scenes `oim.tasks.pusht.PushT` can load.

Each `SceneSpec` supplies everything a scene needs: the MJCF scene file per
embodiment, the object's goal pose, its obstacle field, its footprint
parameterization, and (for xArm6) the arm's ground-mount placement. `PushT`
itself never branches on a scene name or holds scene-specific data -- it
looks up one `SceneSpec` by name and wraps cost functions/ADMM plumbing
around whatever it's handed. A new environment is one new entry here plus
its own MJCF (with its own `<camera>`/`<keyframe>` for recording/starting
pose, read generically by examples/pusht.py), not a change to
oim/tasks/pusht.py.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

import jax.numpy as jnp

from oim.objects import Box, Circle, ObstacleField, Polygon, t_shape_footprint


@dataclass(frozen=True)
class SceneSpec:
    """One scene: which MJCF per embodiment, and the object's own facts.

    Args:
        mjcf_by_robot: Scene path (relative to `oim/models/`) for each
            embodiment this scene supports, e.g.
            `{"point": "pusht_clutter/scene.xml"}`. An embodiment not
            listed here isn't available for this scene.
        goal: Goal pose for the object, world-frame SE(2).
        obstacles: Static obstacles, matching the obstacle geoms in the
            MJCF.
        footprint_kwargs: Passed to `oim.objects.t_shape_footprint()` --
            empty means the default (clutter-scene) T-block size.
        xarm6_base_pos: Ground-mounted (x, y) base placement, xArm6 only.
        xarm6_base_yaw_deg: Base yaw about z (degrees), xArm6 only.
    """

    mjcf_by_robot: Dict[str, str]
    goal: jnp.ndarray
    obstacles: ObstacleField
    footprint_kwargs: Dict[str, object] = field(default_factory=dict)
    xarm6_base_pos: Optional[Tuple[float, float]] = None
    xarm6_base_yaw_deg: Optional[float] = None

    def mjcf_scene(self, robot: str) -> str:
        """Scene path (relative to `oim/models/`) for `robot`.

        Raises:
            ValueError: If this scene has no MJCF for `robot`.
        """
        if robot not in self.mjcf_by_robot:
            raise ValueError(
                f"robot={robot!r} has no scene here (available: "
                f"{sorted(self.mjcf_by_robot)})"
            )
        return self.mjcf_by_robot[robot]

    def footprint(self) -> Polygon:
        """The object's footprint outline for this scene."""
        return t_shape_footprint(**self.footprint_kwargs)


SCENES: Dict[str, SceneSpec] = {
    "clutter": SceneSpec(
        mjcf_by_robot={
            "point": "pusht_clutter/scene.xml",
            "xarm6": "xarm6_pusht_clutter/scene.xml",
        },
        # Goal pose (world-frame SE(2)), matching the `goal` mocap body in
        # models/pusht_clutter/pusht_clutter.xml (shared verbatim by the
        # xarm6 scene).
        goal=jnp.array([0.50, 0.48, jnp.pi / 4]),
        # Static obstacles, matching the obstacle geoms in the same MJCF.
        obstacles=ObstacleField(
            [
                Circle(center=[0.08, 0.32], radius=0.04),
                Box(
                    center=[0.38, 0.10], half_extents=[0.04, 0.035],
                    angle=0.25,
                ),
                Polygon(
                    jnp.array([[0.10, 0.42], [0.20, 0.42], [0.15, 0.52]])
                ),
            ]
        ),
        # Ground-mounted, chosen via the reach sweep in
        # models/xarm6_pusht_clutter/verify_reach.py; covers the
        # block/goal/obstacle footprint within a few cm.
        xarm6_base_pos=(0.2, 0.75),
        xarm6_base_yaw_deg=-90.0,
    ),
    "gym2": SceneSpec(
        # Converted from the Object-Informed-Manipulation (IsaacGym) repo's
        # `conf/task/sim_task02.yaml` ("push the tee block avoiding an
        # obstacle"), matching `models/xarm6_pusht_gym2/`. The T-block's own
        # geometry (footprint_kwargs below) is taken directly from that
        # repo's `assets/urdf/tee_block/tee_block.urdf` -- noticeably bigger
        # than the clutter scene's default footprint.
        #
        # Goal/obstacle/arm-base positions are IsaacGym's own literal
        # world-frame numbers (confirmed identical -- arm base, block
        # start, goal -- across every sim_taskNN.yaml, and via
        # docs/static/results/figures/sim_tasks.png: block starts at
        # (0.7, -0.45), goal at (0.9, 0.30), obstacle at (0.9, 0.05), arm
        # base at (0.4, 0)) *translated* by (-0.7, +0.45) so the block's own
        # start lands at (0, 0). That's required, not a stylistic choice:
        # `PushT._block_pose` reads `qpos` directly for `robot="xarm6"`,
        # which is the T_x/T_y joints' displacement from the block body's
        # MJCF-declared anchor, not the true world position -- every scene
        # keeps that anchor at (0, 0) so the two coincide, and this one
        # must too or every cost term would silently compare the wrong
        # quantity against `goal`. The translation is rigid (preserves
        # every real relative distance/angle from IsaacGym's own numbers,
        # just re-origins them), not a rescale. Goal orientation is
        # IsaacGym's own quat ([0,0,1,0] xyzw), a 180-degree flip about z
        # from the block's spawn orientation (theta=0, matching
        # `t_shape_footprint()`'s own implicit zero), hence `jnp.pi`.
        mjcf_by_robot={"xarm6": "xarm6_pusht_gym2/scene.xml"},
        goal=jnp.array([0.2, 0.75, jnp.pi]),
        obstacles=ObstacleField(
            [Box(center=[0.2, 0.5], half_extents=[0.05, 0.05], angle=0.0)]
        ),
        footprint_kwargs=dict(
            crossbar_half=(0.100, 0.025), stem_half=(0.025, 0.050),
            crossbar_y=0.0, stem_y=-0.075,
        ),
        # IsaacGym's own (0.4, 0), translated by the same (-0.7, +0.45) as
        # everything else above (a rigid shift doesn't change yaw, so
        # that's still 0 -- IsaacGym never rotates its base either, no
        # init_ori on xarm6_stick.yaml).
        xarm6_base_pos=(-0.3, 0.45),
        xarm6_base_yaw_deg=0.0,
    ),
}
