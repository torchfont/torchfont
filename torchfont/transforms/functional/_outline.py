"""Functional whole-outline kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import _ops
from torchfont.transforms.functional._utils import _native_outline

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


def remove_overlaps(
    inpt: Outline, *, verify: bool = False, verify_size: int = 256
) -> Outline:
    """Merge overlapping subpaths.

    Args:
        inpt: Glyph outline to simplify.
        verify: Whether to preserve ``inpt`` when coverage changes.
        verify_size: Verification resolution in pixels. Must be between 1 and
            4096.

    """
    return _native_outline(
        inpt, _ops.remove_overlaps, verify, verify_size, name="remove_overlaps"
    )


def remove_overlap_groups(
    inpt: Outline,
    selection_values: Tensor,
    *,
    verify: bool = False,
    verify_size: int = 256,
) -> Outline:
    """Simplify overlap groups according to explicit selection values.

    Args:
        inpt: Glyph outline whose bbox-connected overlap groups are selected
            for simplification.
        selection_values: Per-group selection values; see
            :class:`~torchfont.transforms.RandomRemoveOverlaps`.
        verify: Whether to preserve ``inpt`` when coverage changes.
        verify_size: Verification resolution in pixels. Must be between 1 and
            4096.

    """
    return _native_outline(
        inpt,
        _ops.remove_overlap_groups,
        selection_values,
        verify,
        verify_size,
        name="remove_overlap_groups",
    )


__all__ = ["remove_overlap_groups", "remove_overlaps"]
