from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import _ops
from torchfont.transforms.functional._utils import _native_outline

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


def normalize_winding(inpt: Outline, *, clockwise: bool = True) -> Outline:
    """Normalize winding while preserving non-zero and even-odd fills.

    Contours whose tight bounding boxes overlap with positive area form groups,
    including containment and transitive overlaps. Contours with disjoint fills
    can belong to the same group when their bounding boxes overlap.
    Each group's largest absolute signed area
    contour determines whether the group is reversed; areas equal within
    floating-point roundoff prefer the first contour. ``clockwise=True`` chooses
    clockwise in y-up font coordinates.
    Groups with open subpaths or only zero-area contours are unchanged.
    Areas within floating-point roundoff are treated as zero.

    Disjoint groups can be reversed independently. Relative winding within a
    group is preserved, so complex groups may retain mixed outer directions.
    Start points may change; contour order and curve geometry are preserved.
    """
    return _native_outline(
        inpt, _ops.normalize_winding, clockwise, name="normalize_winding"
    )


def reverse_winding_groups(inpt: Outline, reversal_mask: Tensor) -> Outline:
    """Reverse winding groups selected by an explicit boolean mask.

    Groups are formed as in :func:`normalize_winding` and ordered by their first
    contour in the input. ``reversal_mask`` must have at least one entry per
    group; extra entries are ignored. Groups containing open subpaths are
    unchanged. Geometry, non-zero fill, and even-odd fill are preserved.
    """
    return _native_outline(
        inpt, _ops.reverse_winding_groups, reversal_mask, name="reverse_winding_groups"
    )


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


__all__ = [
    "normalize_winding",
    "remove_overlap_groups",
    "remove_overlaps",
    "reverse_winding_groups",
]
