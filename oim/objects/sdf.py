"""Signed-distance primitives and obstacle fields for analytic object models.

These are plain JAX functions on planar points, independent of any MuJoCo
model. They exist so that a task's analytic (object-level) subproblem can
describe its environment declaratively instead of hand-rolling geometry
inside the task file.
"""

from abc import ABC, abstractmethod
from typing import Sequence

import jax
import jax.numpy as jnp


def rotate(theta: jax.Array, v: jax.Array) -> jax.Array:
    """Rotate 2D vector(s) `v` of shape (..., 2) by angle(s) `theta`."""
    c, s = jnp.cos(theta), jnp.sin(theta)
    vx, vy = v[..., 0], v[..., 1]
    return jnp.stack([c * vx - s * vy, s * vx + c * vy], axis=-1)


class Shape(ABC):
    """A planar shape exposing a signed distance function."""

    @abstractmethod
    def sdf(self, points: jax.Array) -> jax.Array:
        """Signed distance from each point to the shape (negative inside).

        Args:
            points: Query points of shape (..., 2).

        Returns:
            Signed distances of shape (...,).
        """


class Circle(Shape):
    """A disc."""

    def __init__(self, center: Sequence[float], radius: float) -> None:
        """Set the center and radius."""
        self.center = jnp.asarray(center)
        self.radius = radius

    def sdf(self, points: jax.Array) -> jax.Array:
        """Signed distance to the disc."""
        return jnp.linalg.norm(points - self.center, axis=-1) - self.radius


class Box(Shape):
    """An oriented rectangle."""

    def __init__(
        self,
        center: Sequence[float],
        half_extents: Sequence[float],
        angle: float = 0.0,
    ) -> None:
        """Set the center, half-extents, and orientation."""
        self.center = jnp.asarray(center)
        self.half_extents = jnp.asarray(half_extents)
        self.angle = angle

    def sdf(self, points: jax.Array) -> jax.Array:
        """Signed distance to the oriented box."""
        local = rotate(-self.angle, points - self.center)
        q = jnp.abs(local) - self.half_extents
        outside = jnp.linalg.norm(jnp.clip(q, 0.0, None), axis=-1)
        inside = jnp.clip(jnp.max(q, axis=-1), None, 0.0)
        return outside + inside


class Polygon(Shape):
    """A closed polygon, signed by winding number."""

    def __init__(self, vertices: jax.Array) -> None:
        """Set the vertices, shape (n, 2), in order around the boundary."""
        self.vertices = jnp.asarray(vertices)

    def sdf(self, points: jax.Array) -> jax.Array:
        """Signed distance to the polygon (negative inside)."""
        verts = self.vertices
        n = verts.shape[0]
        best_dist2 = jnp.full(points.shape[:-1], jnp.inf)
        winding = jnp.zeros(points.shape[:-1])
        for i in range(n):
            a, b = verts[i], verts[(i + 1) % n]
            ab = b - a
            t = jnp.clip(
                jnp.sum((points - a) * ab, axis=-1) / jnp.sum(ab * ab), 0.0, 1.0
            )
            proj = a + t[..., None] * ab
            best_dist2 = jnp.minimum(
                best_dist2, jnp.sum((points - proj) ** 2, axis=-1)
            )
            upward = (a[1] <= points[..., 1]) & (b[1] > points[..., 1])
            downward = (a[1] > points[..., 1]) & (b[1] <= points[..., 1])
            is_left = (b[0] - a[0]) * (points[..., 1] - a[1]) - (
                points[..., 0] - a[0]
            ) * (b[1] - a[1])
            winding += jnp.where(upward & (is_left > 0), 1.0, 0.0)
            winding += jnp.where(downward & (is_left < 0), -1.0, 0.0)
        sign = jnp.where(winding != 0, -1.0, 1.0)
        return sign * jnp.sqrt(best_dist2)

    def sample_boundary(self, n_per_edge: int = 4) -> jax.Array:
        """Evenly sample points along the polygon boundary.

        Useful for turning a footprint into a set of collision-check points.

        Args:
            n_per_edge: Number of samples per edge.

        Returns:
            Points of shape (n_vertices * n_per_edge, 2).
        """
        verts = self.vertices
        n = verts.shape[0]
        pts = [
            verts[i] + (verts[(i + 1) % n] - verts[i]) * k / n_per_edge
            for i in range(n)
            for k in range(n_per_edge)
        ]
        return jnp.stack(pts)


class ObstacleField:
    """A collection of static obstacles with a combined clearance cost."""

    def __init__(self, shapes: Sequence[Shape]) -> None:
        """Set the obstacle shapes."""
        self.shapes = list(shapes)

    def sdf(self, points: jax.Array) -> jax.Array:
        """Distance to the nearest obstacle, for each query point."""
        if not self.shapes:
            return jnp.full(points.shape[:-1], jnp.inf)
        return jnp.min(jnp.stack([s.sdf(points) for s in self.shapes]), axis=0)

    def hinge_cost(
        self, points: jax.Array, weight: float, margin: float
    ) -> jax.Array:
        """Squared-hinge clearance penalty, summed over points and obstacles.

        Penalizes any point closer than `margin` to any obstacle.

        Args:
            points: Query points of shape (..., 2).
            weight: Penalty weight.
            margin: Clearance below which the penalty activates.

        Returns:
            The scalar total penalty.
        """
        cost = 0.0
        for shape in self.shapes:
            d = shape.sdf(points)
            cost += weight * jnp.sum(jnp.clip(margin - d, 0.0, None) ** 2)
        return cost
