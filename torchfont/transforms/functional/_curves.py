"""Functional curve conversion and segment kernels."""

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont.structures import Outline
from torchfont.transforms.functional._utils import _native_outline


def _quad_to_cubic(
    types: Tensor, coords: Tensor, *, merge_curves: bool = False
) -> tuple[Tensor, Tensor]:
    """Convert ``ElementType.QUAD_TO`` entries to ``ElementType.CURVE_TO``.

    Accepts a 1-D ``types`` tensor and a 2-D ``coords`` tensor of shape
    ``(N, 6)``. The output has the same shape as the input. Rows in ``coords``
    use the ``[cx0, cy0, cx1, cy1, x, y]`` layout, with quadratic control
    points read from ``[cx0, cy0]`` and endpoints from ``[x, y]``.

    When ``merge_curves=True``, adjacent mergeable curves and lines are merged
    in the same Rust call after conversion. The output length may differ from
    the input in this mode.
    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.quad_to_cubic(
        types.numpy(), coords.reshape(-1).numpy(), merge_curves
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )


def _cubic_to_quad(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Convert ``ElementType.CURVE_TO`` entries to ``ElementType.QUAD_TO`` sequences.

    Each cubic Bezier segment is replaced by the minimum number of quadratic
    Bezier segments needed to approximate it within ~1e-3 em units
    (roughly 1 font-unit in a 1000-UPM font), following the fonttools cu2qu
    approach. Consecutive quadratics share implicit on-curve points at the
    midpoints of adjacent off-curve control points (TrueType spline).

    Unlike ``quad_to_cubic``, the output length may differ from the input
    because a single cubic can expand into multiple quadratics.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new variable-length outline tuple ``(types, coords)`` where each
        ``CurveTo`` has been replaced by one or more ``QuadTo`` path elements.
        Non-cubic path elements are passed through unchanged.

    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.cubic_to_quad(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )


def _merge_curves(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Merge adjacent segments that belong to the same parent curve or line.

    Adjacent cubic and quadratic Bezier segments are merged when they are pieces
    of a single parent curve (i.e. joined at smooth split points determined via
    de Casteljau). Adjacent ``LineTo`` segments are merged when the three points are
    collinear and the segments run in the same direction. Unlike the fonttools
    ``merge_curves`` helper this transform also handles line segments.

    The comparison tolerance is ~1e-3 em units (roughly
    1 font-unit in a 1000-UPM font), matching the precision typically used by
    fonttools.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new variable-length outline tuple ``(types, coords)`` with mergeable
        adjacent segments collapsed.

    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.merge_curves(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )


def quad_to_cubic(inpt: Outline, *, merge_curves: bool = False) -> Outline:
    """Convert quadratic segments to cubic segments."""
    return Outline(*_quad_to_cubic(inpt.types, inpt.coords, merge_curves=merge_curves))


def cubic_to_quad(inpt: Outline) -> Outline:
    """Convert cubic segments to quadratic segments."""
    return Outline(*_cubic_to_quad(inpt.types, inpt.coords))


def merge_curves(inpt: Outline) -> Outline:
    """Merge adjacent pieces of the same parent curve or line."""
    return Outline(*_merge_curves(inpt.types, inpt.coords))


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
        _torchfont.random_split_segments,
        selection_values.cpu().contiguous().numpy(),
        position_values.cpu().contiguous().numpy(),
        split_probability,
        split_range,
    )


__all__ = ["cubic_to_quad", "merge_curves", "quad_to_cubic", "split_segments"]
