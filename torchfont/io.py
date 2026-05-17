"""Shared constants for glyph outline encoding."""

from enum import IntEnum


class ElementType(IntEnum):
    """Integer element types used to encode path elements in outlines."""

    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6


TYPE_DIM: int = len(ElementType)
COORD_DIM: int = 6
