"""Semantic glyph outline structure and encoding constants."""

from dataclasses import dataclass
from enum import IntEnum

import torch
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
_COORD_NDIM = 2


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

    def __post_init__(self) -> None:
        """Reject structurally invalid coupled tensors at construction."""
        if self.types.ndim != 1:
            msg = f"types must be 1-D, got {self.types.ndim}-D"
            raise ValueError(msg)
        if self.coords.ndim != _COORD_NDIM or self.coords.shape[1] != COORD_DIM:
            shape = tuple(self.coords.shape)
            msg = f"coords must have shape (N, {COORD_DIM}), got {shape}"
            raise ValueError(msg)
        if self.types.shape[0] != self.coords.shape[0]:
            msg = "types and coords must have the same number of rows"
            raise ValueError(msg)
        if self.types.dtype is not torch.long:
            msg = f"types must have dtype torch.long, got {self.types.dtype}"
            raise TypeError(msg)
        if self.coords.dtype is not torch.float32:
            msg = f"coords must have dtype torch.float32, got {self.coords.dtype}"
            raise TypeError(msg)
        if self.types.device != self.coords.device:
            msg = "types and coords must be on the same device"
            raise ValueError(msg)


__all__ = ["COORD_DIM", "TYPE_DIM", "ElementType", "Outline"]
