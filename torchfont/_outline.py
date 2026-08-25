"""Semantic glyph outline structure and encoding constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from types import EllipsisType

    _Index = (
        int
        | slice
        | Tensor
        | list[int]
        | tuple[int | slice | Tensor | EllipsisType | None, ...]
        | None
    )


class ElementType(IntEnum):
    """Integer element types used to encode outline path elements."""

    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6


_COORD_DIM = 6


@dataclass(frozen=True, eq=False)
class Outline:
    """Glyph outlines encoded by two coupled tensors.

    ``types`` has shape ``(N,)`` and ``coords`` has shape ``(N, 6)`` with rows
    ``[cx0, cy0, cx1, cy1, x, y]``. Rows correspond
    one-to-one. Coordinates inactive for an element type, including every
    coordinate of ``CLOSE``, ``END``, and ``PAD``, carry no semantic value.

    An ``Outline`` holds references to its tensors rather than copies, so
    mutating ``types`` or ``coords`` in place is visible through it, as tensor
    aliasing normally behaves. Assigning to either attribute raises instead.

    Warning:
        Do not register ``Outline`` as a pytree node. Every transform would
        silently stop doing anything.

    """

    types: Tensor
    coords: Tensor

    def __post_init__(self) -> None:
        """Reject structurally invalid coupled tensors at construction."""
        if self.types.ndim != 1:
            msg = f"types must be 1-D, got {self.types.ndim}-D"
            raise ValueError(msg)
        if (
            self.coords.ndim != self.types.ndim + 1
            or self.coords.shape[1] != _COORD_DIM
        ):
            shape = tuple(self.coords.shape)
            msg = f"coords must have shape (N, {_COORD_DIM}), got {shape}"
            raise ValueError(msg)
        if self.types.shape[0] != self.coords.shape[0]:
            msg = (
                "types shape must match coords without its last dimension, "
                f"got {tuple(self.types.shape)} and {tuple(self.coords.shape)}"
            )
            raise ValueError(msg)
        if self.types.dtype is not torch.long:
            msg = f"types must have dtype torch.long, got {self.types.dtype}"
            raise TypeError(msg)
        if not self.coords.dtype.is_floating_point:
            msg = f"coords must have a floating point dtype, got {self.coords.dtype}"
            raise TypeError(msg)
        if self.types.device != self.coords.device:
            msg = "types and coords must be on the same device"
            raise ValueError(msg)

    @classmethod
    def _wrap(cls, types: Tensor, coords: Tensor) -> Outline:
        """Pair tensors produced by an operation on a valid outline."""
        return cls(types, coords)

    @property
    def shape(self) -> torch.Size:
        """Shape of ``types``, that is ``(N,)``."""
        return self.types.shape

    @property
    def num_elements(self) -> int:
        """Number of path elements."""
        return self.types.shape[-1]

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype of ``coords``."""
        return self.coords.dtype

    @property
    def device(self) -> torch.device:
        """Device shared by ``types`` and ``coords``."""
        return self.coords.device

    def to(
        self,
        device: torch.device | str | int | torch.dtype | None = None,
        dtype: torch.dtype | None = None,
        *,
        non_blocking: bool = False,
    ) -> Outline:
        """Move or cast the outline, following :meth:`torch.Tensor.to`.

        As with tensors, a dtype may be passed as the only positional argument.
        A dtype applies to ``coords`` only and must be floating point, because
        ``types`` is always ``torch.long``.
        """
        if isinstance(device, torch.dtype):
            if dtype is not None:
                msg = "dtype was given both positionally and by keyword"
                raise TypeError(msg)
            device, dtype = None, device
        if dtype is not None and not dtype.is_floating_point:
            msg = f"dtype must be floating point, got {dtype}"
            raise TypeError(msg)
        return self._wrap(
            self.types.to(device=device, non_blocking=non_blocking),
            self.coords.to(device=device, dtype=dtype, non_blocking=non_blocking),
        )

    def pin_memory(self) -> Outline:
        """Return this outline with both tensors in pinned memory."""
        return self._wrap(self.types.pin_memory(), self.coords.pin_memory())

    def __len__(self) -> int:
        """Size of the first dimension, following :func:`len` on a tensor."""
        return self.types.shape[0]

    def __getitem__(self, index: _Index) -> Outline:
        """Index logical outline dimensions, keeping the coordinate axis intact."""
        coord_index = (
            (*index, slice(None)) if isinstance(index, tuple) else (index, slice(None))
        )
        types = self.types[index]
        coords = self.coords[coord_index]
        if types.ndim == 0:
            msg = "indexing must preserve the outline element dimension"
            raise IndexError(msg)
        return self._wrap(types, coords)

    def __repr__(self) -> str:
        """Summarise shape, dtype, and device instead of dumping both tensors."""
        return (
            f"Outline(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device})"
        )


__all__ = [
    "ElementType",
    "Outline",
]
