from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import _ops
from torchfont.transforms.functional._utils import _native_outline

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


def quad_to_cubic(inpt: Outline, *, merge_curves: bool = False) -> Outline:
    """Convert ``QUAD_TO`` elements to ``CURVE_TO`` elements.

    Each quadratic segment maps exactly onto one cubic segment, so the output
    length matches the input unless ``merge_curves`` is enabled.

    Args:
        inpt: Outline to convert.
        merge_curves: Merge adjacent mergeable curves and lines after conversion.
            The output length may then differ from the input.

    """
    return _native_outline(
        inpt,
        _ops.quad_to_cubic,
        merge_curves,
        name="quad_to_cubic",
    )


def cubic_to_quad(inpt: Outline) -> Outline:
    """Convert ``CURVE_TO`` elements to sequences of ``QUAD_TO`` elements.

    Each cubic Bezier segment is approximated by quadratic Bezier segments
    within ~1e-3 em units (roughly one font unit in a 1000-UPM font).

    Unlike :func:`quad_to_cubic`, the output length may differ from the input
    because one cubic can expand into several quadratics.
    """
    return _native_outline(inpt, _ops.cubic_to_quad, name="cubic_to_quad")


def merge_curves(inpt: Outline) -> Outline:
    """Merge adjacent pieces of the same parent curve or line.

    Adjacent cubic and quadratic Bezier segments are merged when they are pieces
    of a single parent curve. Adjacent ``LINE_TO`` segments are merged when
    the three points are collinear and the segments run in the same direction.

    The comparison tolerance is ~1e-3 em units, roughly one font unit in a
    1000-UPM font.
    """
    return _native_outline(inpt, _ops.merge_curves, name="merge_curves")


def split_segments(
    inpt: Outline,
    selection_values: Tensor,
    position_values: Tensor,
    *,
    split_probability: float,
    split_range: tuple[float, float],
) -> Outline:
    """Split segments according to explicit selection and position values."""
    return _native_outline(
        inpt,
        _ops.split_segments,
        selection_values,
        position_values,
        split_probability,
        list(split_range),
        name="split_segments",
    )


__all__ = ["cubic_to_quad", "merge_curves", "quad_to_cubic", "split_segments"]
