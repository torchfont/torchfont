"""Functional subpath operations.

Subpath boundaries are derived from path element types rather than stored
alongside the outline. Operations here reorder or re-encode elements in Rust,
so they do not define a gradient.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchfont import _ops, _torchfont
from torchfont._outline import _COORD_DIM, Outline
from torchfont.transforms.functional._utils import _native_outline, _require_no_grad

if TYPE_CHECKING:
    from torch import Tensor


def split_subpaths(inpt: Outline) -> tuple[Outline, ...]:
    """Split an outline into independently encoded subpaths.

    Each result starts with ``MOVE_TO`` and ends with a newly added ``END``.
    Subpath order, winding, curve degree, coordinates, and whether the subpath
    is open or closed are preserved. An outline with no subpaths returns an
    empty tuple.

    The number of results depends on tensor values, so this operation is not
    supported inside :func:`torch.compile`.
    """
    _require_no_grad(inpt, "split_subpaths")
    arrays = _torchfont.split_subpaths(
        inpt.types.detach().contiguous().numpy(),
        inpt.coords.detach().contiguous().reshape(-1).numpy(),
    )
    return tuple(
        Outline._wrap(  # noqa: SLF001
            torch.from_numpy(types), torch.from_numpy(coords).view(-1, _COORD_DIM)
        )
        for types, coords in arrays
    )


def drop_subpaths(
    inpt: Outline,
    drop_mask: Tensor,
) -> Outline:
    """Drop subpaths selected by an explicit boolean mask."""
    return _native_outline(
        inpt,
        _ops.drop_subpaths,
        drop_mask,
        name="drop_subpaths",
    )


def normalize_subpath_start_points(inpt: Outline) -> Outline:
    """Choose a deterministic start point for each closed subpath.

    Each subpath start moves to its lexicographically smallest ``(x, y)``
    endpoint. Open subpaths, ``END``, and ``PAD`` elements are unchanged. When
    rotation crosses the old closing edge, that implicit edge is materialised as
    ``LINE_TO`` so the represented geometry is preserved.
    """
    return _native_outline(
        inpt,
        _ops.normalize_subpath_start_points,
        name="normalize_subpath_start_points",
    )


def set_subpath_start_points(inpt: Outline, selection_values: Tensor) -> Outline:
    """Set closed-subpath start points from explicit unit-interval values."""
    return _native_outline(
        inpt,
        _ops.set_subpath_start_points,
        selection_values,
        name="set_subpath_start_points",
    )


def reorder_subpaths(inpt: Outline, keys: Tensor) -> Outline:
    """Order subpaths by explicit sort keys."""
    return _native_outline(
        inpt,
        _ops.reorder_subpaths,
        keys,
        name="reorder_subpaths",
    )


__all__ = [
    "drop_subpaths",
    "normalize_subpath_start_points",
    "reorder_subpaths",
    "set_subpath_start_points",
    "split_subpaths",
]
