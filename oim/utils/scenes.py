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

Nothing checks at runtime that a `SceneSpec` and its MJCF agree -- the
planner would simply reason about a world the simulator is not running.
`tests/test_scenes.py` does check it, geom by geom, for every scene here.
"""

from dataclasses import dataclass, field
from typing import Callable, Dict, Optional, Sequence, Tuple

import jax.numpy as jnp

from oim.objects import (
    Box,
    Circle,
    ObstacleField,
    Polygon,
    Shape,
    c_shape_footprint,
    t_shape_footprint,
)


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
        footprint_kwargs: Passed to `footprint_builder` -- empty means that
            builder's own defaults.
        footprint_builder: Builds the pushed object's outline. Defaults to
            `t_shape_footprint`; a scene whose object is not a T (e.g.
            `icra_sign`) names its own.
        xarm6_base_pos: Ground-mounted (x, y) base placement, xArm6 only.
        xarm6_base_yaw_deg: Base yaw about z (degrees), xArm6 only.
    """

    mjcf_by_robot: Dict[str, str]
    goal: jnp.ndarray
    obstacles: ObstacleField
    footprint_kwargs: Dict[str, object] = field(default_factory=dict)
    footprint_builder: Callable[..., Polygon] = t_shape_footprint
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
        return self.footprint_builder(**self.footprint_kwargs)


# ----------------------------------------------------------------------
# The five tabletop pushing scenes
# ----------------------------------------------------------------------
#
# Converted from the Object-Informed-Manipulation (IsaacGym) repo's
# `conf/task/sim_task01..05.yaml`, matching `models/xarm6_pusht_tabletop/`.
# Positions are IsaacGym's own world-frame numbers, rigidly re-origined in
# xy so the pushed object starts at (0, 0) -- required, because
# `PushT._block_pose` reads `qpos`, the object joints' displacement from
# their body's MJCF anchor, so that anchor has to be the origin or every
# cost term compares the wrong quantity. The shift preserves every real
# relative distance and angle. See
# `models/xarm6_pusht_tabletop/common.xml` for the full derivation,
# including the z convention.
#
# The four T-block scenes share the shift (-0.7, +0.45) (tee_block starts
# at (0.7, -0.45)), the same arm base, block and goal -- they differ *only*
# in obstacles, exactly as sim_task01..04 do. `icra_sign` pushes a letter
# instead and shifts by (-0.7, +0.40).

# IsaacGym's assets/urdf/tee_block/tee_block.urdf (crossbar box
# "0.2 0.05 0.05", stem "0.05 0.1 0.05" at y = -0.075; URDF gives full
# extents, these are halves). Noticeably bigger than the clutter scene's T.
_TABLETOP_TEE_FOOTPRINT = dict(
    crossbar_half=(0.100, 0.025),
    stem_half=(0.025, 0.050),
    crossbar_y=0.0,
    stem_y=-0.075,
)

# conf/actors/block.yaml: a 0.1 m cube at (0.9, 0.05), shared by
# `single_obstacle` and `ycb_clutter`.
_TABLETOP_CUBE = Box(center=[0.2, 0.5], half_extents=[0.05, 0.05])


def _tee_scene(name: str, obstacles: Sequence[Shape]) -> SceneSpec:
    """A `SceneSpec` for one of the four T-block scenes.

    They differ only in obstacles, so everything else is set here once.

    Args:
        name: The scene's MJCF basename under
            `models/xarm6_pusht_tabletop/`.
        obstacles: That scene's static obstacles.

    Returns:
        The scene spec.
    """
    return SceneSpec(
        mjcf_by_robot={"xarm6": f"xarm6_pusht_tabletop/{name}.xml"},
        # IsaacGym's goal (0.9, 0.30) with quat [0,0,1,0] (xyzw), a
        # 180-degree flip about z from the block's spawn orientation
        # (theta = 0, matching `t_shape_footprint`'s own implicit zero).
        goal=jnp.array([0.2, 0.75, jnp.pi]),
        obstacles=ObstacleField(list(obstacles)),
        footprint_kwargs=dict(_TABLETOP_TEE_FOOTPRINT),
        # IsaacGym's own (0.4, 0) arm base. A rigid shift doesn't change
        # yaw, and IsaacGym never rotates the base (no init_ori on
        # conf/actors/xarm6_stick.yaml), so that stays 0.
        xarm6_base_pos=(-0.3, 0.45),
        xarm6_base_yaw_deg=0.0,
    )


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
    # sim_task01: "push the tee block". Nothing in the way.
    "open_table": _tee_scene("open_table", []),
    # sim_task02: "... avoiding an obstacle".
    "single_obstacle": _tee_scene("single_obstacle", [_TABLETOP_CUBE]),
    # sim_task03: "... avoiding two shelves". conf/actors/shelf_{1,2}.yaml:
    # 0.2 x 0.25 x 0.2 m boxes at (1.1, 0.05) and (0.7, 0.05). The gap
    # between them is x in [0.1, 0.3] -- exactly as wide as the T's
    # crossbar is long.
    "shelf_gap": _tee_scene(
        "shelf_gap",
        [
            Box(center=[0.0, 0.5], half_extents=[0.10, 0.125]),
            Box(center=[0.4, 0.5], half_extents=[0.10, 0.125]),
        ],
    ),
    # sim_task04: "... avoiding multiple obstacles". The three YCB actors
    # are their meshes' own bounding boxes, placed at the bbox centre (the
    # YCB meshes are not centred on their link origin) -- see
    # ycb_clutter.xml for why boxes rather than meshes.
    "ycb_clutter": _tee_scene(
        "ycb_clutter",
        [
            _TABLETOP_CUBE,
            # spamCan: bbox 0.1021 x 0.0601, centre (0.5672, 0.2734).
            Box(center=[-0.1328, 0.7234], half_extents=[0.0511, 0.0301]),
            # dominoSugar: a URDF box, 0.06 x 0.095 x 0.175, laid on its
            # side by init_ori (-90 degrees about y) so its footprint is
            # 0.175 x 0.095.
            Box(center=[0.2, 0.25], half_extents=[0.0875, 0.0475]),
            # mustardBottle: bbox 0.0972 x 0.0666, centre (1.1847, 0.0765).
            Box(center=[0.4847, 0.5265], half_extents=[0.0486, 0.0333]),
        ],
    ),
    "icra_sign": SceneSpec(
        # sim_task05, respelled: seven fixed glyphs spell "ICRA 2026" in a
        # row at x = 0.2 with the C's own slot left empty, and the goal is
        # that slot. See icra_sign.xml for why the C is the pushed letter.
        mjcf_by_robot={"xarm6": "xarm6_pusht_tabletop/icra_sign.xml"},
        # The empty slot, second from the top of the row: (0.9, 0.45) in
        # IsaacGym's coordinates. Orientation is the source's own glyph
        # quat, [0,0,0.7071,-0.7071] (xyzw) = -90 degrees about z. The C
        # spawns unrotated, so this task needs a quarter turn too.
        goal=jnp.array([0.2, 0.85, -jnp.pi / 2]),
        # Each glyph is the bounding box of its own `*_centered.ply`
        # (scale 0.001), rotated -90 degrees about z like the MJCF geoms:
        # half-extents stay in the glyph's own frame and `angle` carries
        # the rotation, exactly as `<geom euler="0 0 -90">` does. The A
        # reuses R's box; the source assets have no A mesh.
        obstacles=ObstacleField(
            [
                Box([0.2, 1.00], [0.01216, 0.05000], angle=-jnp.pi / 2),  # I
                Box([0.2, 0.70], [0.03563, 0.05000], angle=-jnp.pi / 2),  # R
                Box([0.2, 0.55], [0.03563, 0.05000], angle=-jnp.pi / 2),  # A
                Box([0.2, 0.30], [0.03338, 0.05076], angle=-jnp.pi / 2),  # 2
                Box([0.2, 0.15], [0.03393, 0.05152], angle=-jnp.pi / 2),  # 0
                Box([0.2, 0.00], [0.03338, 0.05076], angle=-jnp.pi / 2),  # 2
                Box([0.2, -0.15], [0.03295, 0.05152], angle=-jnp.pi / 2),  # 6
            ]
        ),
        # The block-letter C's own dimensions, matching icra_sign.xml's
        # three geoms.
        footprint_builder=c_shape_footprint,
        footprint_kwargs=dict(
            half_width=0.0350, half_height=0.0515, half_stroke=0.010
        ),
        # IsaacGym's (0.4, 0), shifted by this scene's own (-0.7, +0.40).
        xarm6_base_pos=(-0.3, 0.40),
        xarm6_base_yaw_deg=0.0,
    ),
}
