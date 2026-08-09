"""Functional whole-outline kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import _ops
from torchfont.transforms.functional._utils import _native_outline

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


def remove_overlaps(inpt: Outline) -> Outline:
    """Merge overlapping subpaths using Skia PathOps winding simplification.

    If PathOps cannot simplify an otherwise valid outline, the original outline
    is returned unchanged.
    """
    return _native_outline(inpt, _ops.remove_overlaps, name="remove_overlaps")


def remove_overlap_groups(inpt: Outline, selection_values: Tensor) -> Outline:
    """Simplify overlap groups according to explicit selection values."""
    return _native_outline(
        inpt,
        _ops.remove_overlap_groups,
        selection_values,
        name="remove_overlap_groups",
    )


__all__ = ["remove_overlap_groups", "remove_overlaps"]
