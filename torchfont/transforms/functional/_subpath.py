"""Functional subpath kernels.

Every kernel here reorders or re-encodes path elements in Rust, so none of them
define a gradient. Subpath boundaries are derived from the ``CLOSE`` and ``END``
element types rather than stored alongside the outline.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import _ops
from torchfont.transforms.functional._utils import _native_outline

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


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
    "normalize_subpath_start_points",
    "reorder_subpaths",
    "set_subpath_start_points",
]
