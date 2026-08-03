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
    """One variable-length glyph outline encoded by two coupled tensors.

    ``types`` has shape ``(N,)`` and ``coords`` has shape ``(N, 6)`` with rows
    ``[cx0, cy0, cx1, cy1, x, y]``. Rows correspond one-to-one. Coordinates
    inactive for an element type, including every coordinate of ``CLOSE``,
    ``END``, and ``PAD``, carry no semantic value. Producers keep both tensors
    on the same device; an ``Outline`` is not a batched glyph representation.
    """

    types: Tensor
    coords: Tensor


__all__ = ["COORD_DIM", "TYPE_DIM", "ElementType", "Outline"]
