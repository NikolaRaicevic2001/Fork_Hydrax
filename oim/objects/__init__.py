"""Analytic (simulator-free) object models for ADMM object-level subproblems.

A `ConsensusTask` describes its object-level subproblem by composing these
pieces -- geometry (`sdf`) and a dynamics + cost model (`planar_pushing`) --
rather than re-deriving them in the task file.
"""

from .planar_pushing import (
    PlanarPushingObject,
    se2_distance_sq,
    t_shape_footprint,
    wrap_angle,
)
from .sdf import Box, Circle, ObstacleField, Polygon, Shape, rotate

__all__ = [
    "Box",
    "Circle",
    "ObstacleField",
    "PlanarPushingObject",
    "Polygon",
    "Shape",
    "rotate",
    "se2_distance_sq",
    "t_shape_footprint",
    "wrap_angle",
]
