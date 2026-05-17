"""Shared constants for glyph outline path element encoding."""

from enum import IntEnum


class ElementType(IntEnum):
    """Integer path element IDs emitted in glyph outline sequences."""

    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6


TYPE_DIM: int = len(ElementType)
COORD_DIM: int = 6
