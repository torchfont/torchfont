"""Shared constants for glyph outline command encoding."""

from enum import IntEnum


class CommandType(IntEnum):
    """Integer command IDs emitted in glyph outline sequences."""

    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6


TYPE_DIM: int = len(CommandType)
COORD_DIM: int = 6
METRICS_DIM: int = 6
"""Number of scalar values in a glyph metrics vector.

The metrics vector layout is ``[advance_width, lsb, x_min, y_min, x_max, y_max]``,
all normalized by ``units_per_em``. ``lsb`` is the left side bearing. Bounding
box values are derived from the convex hull of all outline control points.
"""

__all__ = ["COORD_DIM", "METRICS_DIM", "TYPE_DIM", "CommandType"]
