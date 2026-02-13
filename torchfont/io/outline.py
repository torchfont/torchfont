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

__all__ = ["COORD_DIM", "TYPE_DIM", "CommandType"]
