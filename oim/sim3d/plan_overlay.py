"""Live/recorded overlay for ADMM: trajectories drawn into the MuJoCo scene.

Draws, for both the object and the robot block, the sampled candidate
trajectories it considered and the one it settled on (the mean-weighted
result of its MPPI-style update, not literally one of the samples). Both
kinds of trajectory are drawn identically -- same line style, same fading
alpha -- the only difference is color: `SAMPLE_COLOR` (light green) for a
candidate, `CHOSEN_COLOR` (bright orange) for the one taken. Object and
robot are told apart by where their paths sit, not by color.
"""

from typing import Optional, Sequence

import mujoco
import numpy as np

SAMPLE_COLOR = (0.55, 0.90, 0.35)  # light green: a rollout that was tried
CHOSEN_COLOR = (1.0, 0.55, 0.0)  # bright orange: the trajectory taken


class PlanOverlay:
    """Draws both blocks' candidate and chosen trajectories into an `MjvScene`.

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
        width: float = 2.5,
        alpha_near: float = 0.95,
        alpha_far: float = 0.35,
        object_height: float = 0.055,
        max_samples: int = 16,
    ) -> None:
        """Configure the overlay's geometry.

        Args:
            horizon: Number of predicted poses per plan, H.
            width: Line width of every path, in pixels -- one width for
                both chosen and sampled paths, since they are drawn the
                same way.
            alpha_near: Opacity at the start of the horizon.
            alpha_far: Opacity at the end of it, so time direction reads
                off the image without needing an animation.
            object_height: World z to draw the object's paths at. The
                default clears the block's top face. The robot's paths are
                drawn at their real height instead, since they are genuine
                3D positions, not lifted SE(2) poses.
            max_samples: Draw at most this many sampled rollouts per block,
                evenly spaced through the population -- drawing all of
                them would be unreadable and slow.
        """
        self.horizon = horizon
        self.width = width
        self.alpha_near = alpha_near
        self.alpha_far = alpha_far
        self.object_height = object_height
        self.max_samples = max_samples

        # One path: a segment between each pair of consecutive poses. Some
        # paths passed to `draw` have H poses, some H+1 (a rollout's own
        # final state, appended where it's cheap to); H segments is a safe
        # upper bound either way.
        self.per_path = horizon

    @property
    def geom_count(self) -> int:
        """Scene geoms one `draw` call consumes at most, for reserving space."""
        # A chosen path each for object and robot, plus up to `max_samples`
        # sampled paths each.
        return 2 * self.per_path + 2 * self.max_samples * self.per_path

    def _alpha(self, k: int, n: int, color: Sequence[float]) -> np.ndarray:
        """Color, faded from `alpha_near` to `alpha_far` over `0..n-1`."""
        frac = k / max(n - 1, 1)
        alpha = self.alpha_near + frac * (self.alpha_far - self.alpha_near)
        return np.array([*color, alpha], dtype=np.float64)

    @staticmethod
    def _init_line(scene: mujoco.MjvScene, i: int) -> None:
        mujoco.mjv_initGeom(
            scene.geoms[i],
            type=mujoco.mjtGeom.mjGEOM_LINE,
            size=np.zeros(3),
            pos=np.zeros(3),
            mat=np.eye(3).flatten(),
            rgba=np.zeros(4),
        )

    def _draw_path(
        self,
        scene: mujoco.MjvScene,
        start: int,
        points: np.ndarray,
        color: Sequence[float],
    ) -> int:
        """One path, as connected line segments. Returns geoms consumed."""
        n = len(points)
        i = start
        for k in range(n - 1):
            self._init_line(scene, i)
            scene.geoms[i].rgba[:] = self._alpha(k, n, color)
            mujoco.mjv_connector(
                scene.geoms[i],
                mujoco.mjtGeom.mjGEOM_LINE,
                self.width,
                points[k],
                points[k + 1],
            )
            i += 1
        return i - start

    def _draw_samples(
        self, scene: mujoco.MjvScene, start: int, samples: Optional[np.ndarray]
    ) -> int:
        """Up to `max_samples` of a block's sampled paths; geoms used."""
        if samples is None or len(samples) == 0:
            return 0
        samples = np.asarray(samples)
        n_show = min(len(samples), self.max_samples)
        idx = np.linspace(0, len(samples) - 1, n_show).astype(int)
        i = start
        for s in idx:
            i += self._draw_path(scene, i, samples[s, :, :3], SAMPLE_COLOR)
        return i - start

    def _lift(self, poses: np.ndarray) -> np.ndarray:
        """An SE(2) pose sequence's (x, y), at this overlay's object height."""
        xy = np.asarray(poses)[:, :2]
        return np.concatenate(
            [xy, np.full((len(xy), 1), self.object_height)], axis=1
        )

    def draw(
        self,
        scene: mujoco.MjvScene,
        object_plan: np.ndarray,
        robot_trace: np.ndarray,
        object_samples: Optional[np.ndarray] = None,
        robot_samples: Optional[np.ndarray] = None,
        draw_chosen: bool = True,
        base: Optional[int] = None,
    ) -> None:
        """Draw both blocks' sampled and chosen trajectories.

        Args:
            scene: The scene to write into -- a viewer's `user_scn` or a
                `mujoco.Renderer`'s `.scene`.
            object_plan: The object block's chosen predicted poses,
                (H, 3), SE(2).
            robot_trace: The robot block's chosen end-effector positions
                along its planned controls, (H, 3), world xyz.
            object_samples: The object block's sampled candidate
                trajectories, (num_samples, H, 3), SE(2). `None` skips
                drawing them.
            robot_samples: The robot block's sampled rollouts' end-effector
                positions, (num_samples, H, 3) or (num_samples, H+1, 3).
                `None` skips drawing them.
            draw_chosen: Whether to draw `object_plan`/`robot_trace` at
                all -- `False` shows only the sample clouds.
            base: First geom index to write. Pass a fixed value for a
                persistent scene, so the overlay keeps its own slot and
                leaves earlier geoms (e.g. `show_traces`) alone. Omit for a
                scene `update_scene` has just rebuilt, to append after the
                model's own geoms.

        Raises:
            ValueError: If `object_plan` or `robot_trace` is not the
                horizon this overlay was built for.
            RuntimeError: If the scene has too few free geoms.
        """
        for name, plan in (
            ("object_plan", object_plan),
            ("robot_trace", robot_trace),
        ):
            if len(plan) != self.horizon:
                raise ValueError(
                    f"{name} has {len(plan)} poses, overlay was built for "
                    f"{self.horizon}"
                )
        start = scene.ngeom if base is None else base
        if start + self.geom_count > scene.maxgeom:
            raise RuntimeError(
                f"plan overlay needs up to {self.geom_count} scene geoms, "
                f"only {scene.maxgeom - start} free"
            )
        i = start

        if object_samples is not None:
            lifted = np.stack(
                [self._lift(p) for p in np.asarray(object_samples)]
            )
            i += self._draw_samples(scene, i, lifted)
        i += self._draw_samples(scene, i, robot_samples)

        if draw_chosen:
            obj_pts = self._lift(object_plan)
            i += self._draw_path(scene, i, obj_pts, CHOSEN_COLOR)
            robot_pts = np.asarray(robot_trace)[:, :3]
            i += self._draw_path(scene, i, robot_pts, CHOSEN_COLOR)

        scene.ngeom = max(scene.ngeom, i)
