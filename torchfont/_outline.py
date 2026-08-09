"""Semantic glyph outline structure and encoding constants."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    from collections.abc import Sequence
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


TYPE_DIM: int = len(ElementType)
COORD_DIM: int = 6


@dataclass(frozen=True, eq=False)
class Outline:
    """Glyph outlines encoded by two coupled tensors.

    ``types`` has shape ``(*batch, N)`` and ``coords`` has shape
    ``(*batch, N, 6)`` with rows ``[cx0, cy0, cx1, cy1, x, y]``. Rows correspond
    one-to-one. Coordinates inactive for an element type, including every
    coordinate of ``CLOSE``, ``END``, and ``PAD``, carry no semantic value.

    A single glyph has an empty ``batch`` shape; :func:`pad_outlines` stacks
    single glyphs into a batch padded with ``ElementType.PAD``. Most transforms
    operate on single glyphs only and reject batches explicitly.

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
        if self.types.ndim < 1:
            msg = f"types must have at least 1 dimension, got {self.types.ndim}"
            raise ValueError(msg)
        if self.coords.ndim != self.types.ndim + 1:
            msg = (
                f"coords must have exactly one more dimension than types, "
                f"got {self.coords.ndim} and {self.types.ndim}"
            )
            raise ValueError(msg)
        if self.coords.shape[-1] != COORD_DIM:
            shape = tuple(self.coords.shape)
            msg = f"coords must have shape (*batch, N, {COORD_DIM}), got {shape}"
            raise ValueError(msg)
        if self.types.shape != self.coords.shape[:-1]:
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
        """Pair tensors already known to satisfy every structural invariant.

        Kernels that derive an outline from a validated outline use this to
        avoid re-checking invariants they cannot have broken. Validation stays
        at the boundary where untrusted tensors enter, in ``__post_init__``.
        """
        outline = cls.__new__(cls)
        object.__setattr__(outline, "types", types)
        object.__setattr__(outline, "coords", coords)
        return outline

    @property
    def shape(self) -> torch.Size:
        """Shape of ``types``, that is ``(*batch, N)``."""
        return self.types.shape

    @property
    def batch_shape(self) -> torch.Size:
        """Leading batch dimensions, empty for a single glyph."""
        return torch.Size(self.types.shape[:-1])

    @property
    def num_elements(self) -> int:
        """Number of path elements per glyph, including padding."""
        return self.types.shape[-1]

    @property
    def is_batched(self) -> bool:
        """Whether this outline carries at least one batch dimension."""
        return self.types.ndim > 1

    @property
    def dtype(self) -> torch.dtype:
        """Floating point dtype of ``coords``."""
        return self.coords.dtype

    @property
    def device(self) -> torch.device:
        """Device shared by ``types`` and ``coords``."""
        return self.coords.device

    @property
    def padding_mask(self) -> Tensor:
        """Boolean mask that is ``True`` where an element is ``PAD``.

        Pass this to attention modules expecting ``key_padding_mask``.
        """
        return self.types == ElementType.PAD.value

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

    def unbind(self) -> tuple[Outline, ...]:
        """Split the first batch dimension without changing its contents."""
        if not self.is_batched:
            msg = "unbind requires a batched outline"
            raise ValueError(msg)
        return tuple(self[index] for index in range(self.types.shape[0]))

    def _strip_padding(self) -> Outline:
        """Drop trailing ``PAD`` elements along the last dimension."""
        if self.is_batched:
            msg = "padding can only be stripped from a single outline"
            raise ValueError(msg)
        kept = (self.types != ElementType.PAD.value).nonzero()
        length = 0 if kept.numel() == 0 else int(kept[-1]) + 1
        return self._wrap(self.types[:length], self.coords[:length])

    def __repr__(self) -> str:
        """Summarise shape, dtype, and device instead of dumping both tensors."""
        return (
            f"Outline(shape={tuple(self.shape)}, dtype={self.dtype}, "
            f"device={self.device})"
        )


def pad_outlines(outlines: Sequence[Outline]) -> Outline:
    """Stack single outlines into one batch padded with ``ElementType.PAD``.

    Every input must be a single glyph on a common device with a common
    ``coords`` dtype. Shorter outlines are padded to the longest length with
    ``PAD`` element types and zero coordinates.

    Use :attr:`Outline.padding_mask` on the result to recover which elements are
    padding, and :meth:`Outline.unbind` to undo the padding.
    """
    if len(outlines) == 0:
        msg = "outlines must not be empty"
        raise ValueError(msg)
    first = outlines[0]
    # Only the dtype is checked: a mismatched shape or device makes the copy
    # below raise, while a mismatched dtype would silently cast the coordinates.
    if any(outline.dtype != first.dtype for outline in outlines):
        msg = "all outlines must share one coords dtype"
        raise ValueError(msg)
    length = max(outline.num_elements for outline in outlines)
    types = first.types.new_full((len(outlines), length), ElementType.PAD.value)
    coords = first.coords.new_zeros((len(outlines), length, COORD_DIM))
    for index, outline in enumerate(outlines):
        end = outline.num_elements
        types[index, :end] = outline.types
        coords[index, :end] = outline.coords
    return Outline._wrap(types, coords)  # noqa: SLF001


def unpad_outlines(outline: Outline) -> tuple[Outline, ...]:
    """Split a padded one-dimensional batch and remove trailing padding."""
    if len(outline.batch_shape) != 1:
        msg = (
            "unpad_outlines requires exactly one batch dimension, got "
            f"{tuple(outline.batch_shape)}"
        )
        raise ValueError(msg)
    return tuple(part._strip_padding() for part in outline.unbind())  # noqa: SLF001


__all__ = [
    "COORD_DIM",
    "TYPE_DIM",
    "ElementType",
    "Outline",
    "pad_outlines",
    "unpad_outlines",
]
