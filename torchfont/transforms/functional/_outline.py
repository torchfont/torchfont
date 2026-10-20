from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchfont import _ops
from torchfont.transforms.functional._utils import _native_outline

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


def normalize_winding(inpt: Outline, *, clockwise: bool | Tensor = True) -> Outline:
    """Normalize winding while preserving non-zero and even-odd fills.

    Contours form a group when their individual non-zero filled interiors
    overlap, including containment and transitive overlaps. Touching only at
    points or along shared edges does not join groups. Grouping uses curve
    geometry without rasterization. Overlap tests use floating-point geometry;
    extremely narrow overlaps may be missed, limiting fill preservation at that
    numerical precision.

    Each group's sum of signed contour areas determines its direction.
    Groups whose sum is zero within floating-point roundoff are unchanged.
    ``clockwise=True`` chooses a negative signed area in y-up font coordinates.
    A boolean tensor specifies one direction per group in input order; a single
    entry applies to all groups. Extra entries are ignored.
    Open subpaths are implicitly closed for fill and area calculations;
    their open or closed status is preserved in the output.

    Only whole contours are reversed. Relative winding within a group is
    preserved, so complex groups may retain mixed outer directions.
    Start points may change; contour order and curve geometry are preserved.
    """
    if isinstance(clockwise, bool):
        clockwise = torch.tensor([clockwise])
    return _native_outline(
        inpt, _ops.normalize_winding, clockwise, name="normalize_winding"
    )


def remove_overlaps(
    inpt: Outline, *, verify: bool = False, verify_size: int = 256
) -> Outline:
    """Merge overlapping subpaths.

    Open subpaths are implicitly closed for filling and may become closed.

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
        inpt: Glyph outline whose contours are grouped by overlapping
            non-zero filled interiors for simplification. Point and shared-edge
            contacts alone do not join groups. Single contours are also
            eligible. Open subpaths are implicitly closed for filling and may
            become closed.
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
]
