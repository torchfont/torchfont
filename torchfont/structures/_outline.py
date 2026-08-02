"""Semantic glyph outline structure and encoding constants."""

from dataclasses import dataclass
from enum import IntEnum

from torch import Tensor


class ElementType(IntEnum):
    """Integer element types used to encode outline path elements."""

    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6


TYPE_DIM: int = len(ElementType)
COORD_DIM: int = 6


@dataclass(frozen=True)
class Outline:
    """A glyph outline represented by coupled element-type and coordinate tensors."""

    types: Tensor
    coords: Tensor


__all__ = ["COORD_DIM", "TYPE_DIM", "ElementType", "Outline"]
