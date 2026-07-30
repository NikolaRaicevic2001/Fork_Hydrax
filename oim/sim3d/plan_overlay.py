"""Live viewer overlay for the two ADMM blocks' predicted object motion.

The consensus variable is a *wrench*, which is hard to read as a number.
What it is negotiating over, though, is object motion -- and both blocks
predict a trajectory of the same object, so drawing the pair turns the
primal residual from a scalar into something you can look at: where the two
curves lie on top of each other the blocks agree, and where they peel apart
is precisely where they do not.

Each predicted pose is drawn as a short bar through the object's origin
along its body x-axis, so the overlay shows *orientation* as well as
position. For the push-T footprint that bar is the crossbar, which makes the
predicted heading immediately recognizable -- and orientation is half the
goal in this task, so a position-only path would hide a real class of
disagreement.
"""

from typing import Optional, Sequence, Tuple

import mujoco
import numpy as np

# Amber and teal, rather than the obvious red/green: they differ in
# luminance as well as hue, so they stay distinguishable in a grayscale
# screenshot and for the most common forms of color vision deficiency.
OBJECT_COLOR = (0.98, 0.62, 0.11)  # what the object block intends
ROBOT_COLOR = (0.13, 0.72, 0.75)  # what the robot block would produce


class PlanOverlay:
    """Draws both blocks' predicted object trajectories into an `MjvScene`.

    Holds no scene of its own, because the two scenes it has to serve behave
    differently. The passive viewer's `user_scn` persists between frames, so
    the overlay owns a fixed slot in it. An offscreen `mujoco.Renderer`'s
    scene is rebuilt from the model by every `update_scene` call, which
    discards anything added previously -- so there the overlay must be
    appended again for each frame. `draw` covers both: pass a fixed `base`
    for a persistent scene, or omit it to append to a freshly rebuilt one.
    """

    def __init__(
        self,
        horizon: int,
        height: float = 0.055,
        bar_half_length: float = 0.042,
        bar_stride: int = 3,
        path_width: float = 4.0,
        pose_width: float = 7.0,
        alpha_near: float = 0.95,
        alpha_far: float = 0.25,
    ) -> None:
        """Configure the overlay's geometry and colour ramp.

        Args:
            horizon: Number of predicted poses per plan, H.
            height: World z to draw at. The default clears the block's top
                face, so the plans float above the scene instead of being
                swallowed by the geometry they describe.
            bar_half_length: Half-length of the orientation bar, in metres.
            bar_stride: Draw an orientation bar every this many poses. The
                object moves a couple of centimetres per planning step
                against a bar some centimetres long, so a bar at every pose
                overlaps its neighbours into a solid ladder that hides the
                path it is annotating. The path itself stays full
                resolution; only the bars are thinned.
            path_width: Line width of the connecting path, in pixels.
            pose_width: Line width of the orientation bars, in pixels.
            alpha_near: Opacity at the start of the horizon.
            alpha_far: Opacity at the end of it, so time direction reads
                off the image without needing an animation.
        """
        self.horizon = horizon
        self.height = height
        self.bar_half_length = bar_half_length
        self.path_width = path_width
        self.pose_width = pose_width
        self.alpha_near = alpha_near
        self.alpha_far = alpha_far

        # Which poses get an orientation bar. The last is always included:
        # the end of the horizon is where the two plans differ most, so it
        # is the one pose whose heading you least want thinned away.
        n_poses = horizon + 1  # the current pose is prepended
        self.bar_at = list(range(0, n_poses, max(bar_stride, 1)))
        if self.bar_at[-1] != n_poses - 1:
            self.bar_at.append(n_poses - 1)

        # Per plan: one path segment between consecutive poses, plus the
        # orientation bars.
        self.per_plan = (n_poses - 1) + len(self.bar_at)

    @property
    def geom_count(self) -> int:
        """Scene geoms one `draw` consumes, for callers reserving space."""
        return 2 * self.per_plan

    def _point(self, pose: np.ndarray) -> np.ndarray:
        """Lift an SE(2) pose's position to the overlay's drawing plane."""
        return np.array([pose[0], pose[1], self.height])

    def _bar_ends(self, pose: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """The two ends of the orientation bar for an SE(2) pose."""
        offset = self.bar_half_length * np.array(
            [np.cos(pose[2]), np.sin(pose[2]), 0.0]
        )
        centre = self._point(pose)
        return centre - offset, centre + offset

    def _draw_plan(
        self,
        scene: mujoco.MjvScene,
        start: int,
        poses: np.ndarray,
        color: Sequence[float],
    ) -> None:
        """Write one plan's path and orientation bars from `start` onward."""
        i = start
        n = len(poses)

        def _alpha(k: int) -> np.ndarray:
            frac = k / max(n - 1, 1)
            return np.array(
                [
                    *color,
                    self.alpha_near + frac * (self.alpha_far - self.alpha_near),
                ],
                dtype=np.float64,
            )

        for k in range(n - 1):
            self._init(scene, i)
            scene.geoms[i].rgba[:] = _alpha(k)
            mujoco.mjv_connector(
                scene.geoms[i],
                mujoco.mjtGeom.mjGEOM_LINE,
                self.path_width,
                self._point(poses[k]),
                self._point(poses[k + 1]),
            )
            i += 1

        for k in self.bar_at:
            bar_start, bar_end = self._bar_ends(poses[k])
            self._init(scene, i)
            scene.geoms[i].rgba[:] = _alpha(k)
            mujoco.mjv_connector(
                scene.geoms[i],
                mujoco.mjtGeom.mjGEOM_LINE,
                self.pose_width,
                bar_start,
                bar_end,
            )
            i += 1

    @staticmethod
    def _init(scene: mujoco.MjvScene, i: int) -> None:
        """Reset geom `i` to a blank line before `mjv_connector` fills it."""
        mujoco.mjv_initGeom(
            scene.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.zeros(4),
        )

    def draw(
        self,
        scene: mujoco.MjvScene,
        current_pose: np.ndarray,
        object_plan: np.ndarray,
        robot_plan: np.ndarray,
        base: Optional[int] = None,
    ) -> None:
        """Draw both plans into `scene`.

        Args:
            scene: The scene to write into -- a viewer's `user_scn` or a
                `mujoco.Renderer`'s `.scene`.
            current_pose: The object's pose now, (3,). Prepended to both
                plans so each path starts at the object rather than one
                planning step ahead of it -- otherwise the two curves
                appear to float free of the thing they describe.
            object_plan: The object block's predicted poses, (H, 3).
            robot_plan: The robot block's predicted poses, (H, 3).
            base: First geom index to write. Pass a fixed value for a
                persistent scene, so the overlay keeps its own slot and
                leaves earlier geoms (e.g. `show_traces`) alone. Omit for a
                scene `update_scene` has just rebuilt, to append after the
                model's own geoms.

        Raises:
            ValueError: If either plan is not the horizon this overlay was
                built for.
            RuntimeError: If the scene has too few free geoms.
        """
        for name, plan in (("object", object_plan), ("robot", robot_plan)):
            if len(plan) != self.horizon:
                raise ValueError(
                    f"{name}_plan has {len(plan)} poses, overlay was built "
                    f"for {self.horizon}"
                )
        start = scene.ngeom if base is None else base
        if start + self.geom_count > scene.maxgeom:
            raise RuntimeError(
                f"plan overlay needs {self.geom_count} scene geoms, only "
                f"{scene.maxgeom - start} free"
            )

        now = np.asarray(current_pose)[None, :3]
        obj = np.vstack([now, np.asarray(object_plan)[:, :3]])
        rob = np.vstack([now, np.asarray(robot_plan)[:, :3]])
        self._draw_plan(scene, start, obj, OBJECT_COLOR)
        self._draw_plan(scene, start + self.per_plan, rob, ROBOT_COLOR)
        scene.ngeom = max(scene.ngeom, start + self.geom_count)
