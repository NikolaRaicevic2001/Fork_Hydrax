"""Signed-distance primitives and obstacle fields for analytic object models.

These are plain JAX functions on planar points, independent of any MuJoCo
model. They exist so that a task's analytic (object-level) subproblem can
describe its environment declaratively instead of hand-rolling geometry
inside the task file.
"""

from abc import ABC, abstractmethod
from typing import Sequence, Tuple

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

    # Finite-difference step for `sdf_and_grad`. Small against the geometry
    # (footprints here are ~0.1 m across) but far above float32 noise.
    _grad_eps = 1e-4

    def sdf_and_grad(self, points: jax.Array) -> Tuple[jax.Array, jax.Array]:
        """Signed distance and its unit outward gradient.

        Uses central differences on `sdf` rather than autodiff, so a new
        `Shape` still only has to implement `sdf` -- but, critically, the
        result stays correct *on the boundary itself*, which is exactly
        where contact normals are needed.

        Autodiff fails there. `Polygon.sdf` is `sign * sqrt(dist2)`, and a
        query lying exactly on an edge has `dist2 == 0`, where `sqrt` has an
        unbounded derivative; guarding that (by clamping or softening)
        instead makes the gradient vanish, silently yielding a contact
        normal that is not merely imprecise but *wrong* -- measured 45
        degrees off the true edge normal for a point on the T's crossbar,
        which sent the contact-point search to the wrong face. Since
        `project_to_boundary` deliberately lands on the boundary, that is
        the common case here, not a corner case.

        Central differences have no such problem: on a flat face they
        recover the face normal exactly, and at a vertex they return the
        average of the two adjoining face normals -- a sane representative
        of the normal cone.

        Args:
            points: Query points of shape (..., 2).

        Returns:
            Signed distances of shape (...,) and unit outward gradients of
            shape (..., 2).
        """
        pts = jnp.asarray(points, dtype=float)
        single = pts.ndim == 1
        flat = pts.reshape(-1, 2)

        h = self._grad_eps
        ex, ey = jnp.array([h, 0.0]), jnp.array([0.0, h])
        d = self.sdf(flat)
        gx = self.sdf(flat + ex) - self.sdf(flat - ex)
        gy = self.sdf(flat + ey) - self.sdf(flat - ey)
        grad = jnp.stack([gx, gy], axis=-1) / (2.0 * h)

        # The gradient still genuinely vanishes on the medial axis, where two
        # equidistant faces contribute opposing directions that cancel. That
        # is a real ambiguity rather than a numerical artefact, but a zero
        # "unit normal" would propagate silently into the contact wrench, so
        # fall back to the radial direction from the shape's center --
        # always defined, and outward for any star-shaped footprint.
        norm = jnp.linalg.norm(grad, axis=-1, keepdims=True)
        radial = flat - jnp.reshape(self.center, (1, 2))
        radial_norm = jnp.linalg.norm(radial, axis=-1, keepdims=True)
        fallback = radial / jnp.where(radial_norm > 1e-12, radial_norm, 1.0)
        grad = jnp.where(
            norm > 1e-6, grad / jnp.where(norm > 1e-6, norm, 1.0), fallback
        )
        if single:
            return d[0], grad[0]
        return d.reshape(pts.shape[:-1]), grad.reshape(pts.shape)

    def project_to_boundary(
        self, points: jax.Array, iterations: int = 3
    ) -> jax.Array:
        """Project points onto the shape boundary.

        Iterates the first-order step Pi(p) = p - d(p) grad d(p). One step
        is exact only when the gradient already points at the nearest
        boundary point; for a query further inside a non-convex footprint it
        undershoots, so the step is repeated a fixed (jit-friendly) number
        of times.

        Caveat: a point sitting *on* the medial axis cannot be projected by
        this family of methods at all -- the gradient there is degenerate
        (two equidistant faces cancel), so the iteration stalls. The T
        footprint's own origin is such a point. Callers that need a
        guaranteed-boundary result from an arbitrary interior seed should
        start from `sample_boundary` instead of relying on projection.

        Args:
            points: Query points of shape (..., 2).
            iterations: Number of first-order steps.

        Returns:
            Boundary points of the same shape.
        """
        p = jnp.asarray(points, dtype=float)
        for _ in range(iterations):
            d, grad = self.sdf_and_grad(p)
            p = p - d[..., None] * grad
        return p

    @property
    @abstractmethod
    def center(self) -> jax.Array:
        """A representative interior point, used for broad-phase culling."""

    @property
    @abstractmethod
    def bounding_radius(self) -> float:
        """Radius of a disc about `center` that contains the whole shape."""


class Circle(Shape):
    """A disc."""

    def __init__(self, center: Sequence[float], radius: float) -> None:
        """Set the center and radius."""
        self._center = jnp.asarray(center)
        self.radius = radius

    def sdf(self, points: jax.Array) -> jax.Array:
        """Signed distance to the disc."""
        return jnp.linalg.norm(points - self._center, axis=-1) - self.radius

    @property
    def center(self) -> jax.Array:
        """The disc's center."""
        return self._center

    @property
    def bounding_radius(self) -> float:
        """The disc's own radius."""
        return float(self.radius)


class Box(Shape):
    """An oriented rectangle."""

    def __init__(
        self,
        center: Sequence[float],
        half_extents: Sequence[float],
        angle: float = 0.0,
    ) -> None:
        """Set the center, half-extents, and orientation."""
        self._center = jnp.asarray(center)
        self.half_extents = jnp.asarray(half_extents)
        self.angle = angle

    def sdf(self, points: jax.Array) -> jax.Array:
        """Signed distance to the oriented box."""
        local = rotate(-self.angle, points - self._center)
        q = jnp.abs(local) - self.half_extents
        outside = jnp.linalg.norm(jnp.clip(q, 0.0, None), axis=-1)
        inside = jnp.clip(jnp.max(q, axis=-1), None, 0.0)
        return outside + inside

    @property
    def center(self) -> jax.Array:
        """The box's center."""
        return self._center

    @property
    def bounding_radius(self) -> float:
        """Half-diagonal of the box, so rotation never escapes the disc."""
        return float(jnp.linalg.norm(self.half_extents))


class Capsule(Shape):
    """A stadium: every point within `radius` of the segment from `a` to `b`.

    The 2D counterpart of MuJoCo's `type="capsule"`, which is what the
    xArm6's pushing stick actually is (see `oim/models/xarm6/xarm6.xml`), so
    a 2D model of that end-effector -- or of a rod-shaped obstacle -- has a
    matching primitive rather than an approximating box.
    """

    def __init__(
        self, a: Sequence[float], b: Sequence[float], radius: float
    ) -> None:
        """Set the segment endpoints and the radius."""
        self.a = jnp.asarray(a, dtype=float)
        self.b = jnp.asarray(b, dtype=float)
        self.radius = radius

    def sdf(self, points: jax.Array) -> jax.Array:
        """Distance to the segment, less the radius."""
        pa = points - self.a
        ba = self.b - self.a
        h = jnp.clip(jnp.sum(pa * ba, axis=-1) / jnp.sum(ba * ba), 0.0, 1.0)
        closest = self.a + h[..., None] * ba
        return jnp.linalg.norm(points - closest, axis=-1) - self.radius

    @property
    def center(self) -> jax.Array:
        """Midpoint of the segment."""
        return 0.5 * (self.a + self.b)

    @property
    def bounding_radius(self) -> float:
        """Half the segment length plus the radius."""
        return 0.5 * float(jnp.linalg.norm(self.b - self.a)) + self.radius


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
        # No floor needed under the sqrt: `sqrt(0)` is a perfectly good
        # *value*, and only its derivative is unbounded -- which no longer
        # matters now that `sdf_and_grad` uses central differences rather
        # than differentiating this function.
        return sign * jnp.sqrt(best_dist2)

    @property
    def center(self) -> jax.Array:
        """Centroid of the vertices."""
        return jnp.mean(self.vertices, axis=0)

    @property
    def bounding_radius(self) -> float:
        """Distance from the centroid to the farthest vertex."""
        c = jnp.mean(self.vertices, axis=0)
        return float(jnp.max(jnp.linalg.norm(self.vertices - c, axis=-1)))

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
