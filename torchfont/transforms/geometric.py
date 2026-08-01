"""Geometric transformation functions for glyph outline tensors.

All functions follow the same convention as :mod:`torchfont.transforms`:
they accept ``(types, coords)`` and return a transformed ``(types, coords)``
pair without modifying the inputs.

Coordinates layout (``coords`` shape ``(N, 6)``)::

    [cx0, cy0, cx1, cy1, x, y]

    Pair 0 (cx0, cy0): off-curve control point 1 — active for QUAD_TO / CURVE_TO
    Pair 1 (cx1, cy1): off-curve control point 2 — active for CURVE_TO only
    Pair 2 (x,   y  ): on-curve endpoint        — active for all drawing path elements

All coordinates are in em units: font design units divided by ``unitsPerEm``.
The glyph body typically occupies
``[0, 1] x [0, 1]`` inside the full canvas ``[-0.25, 1.25] x [-0.25, 1.25]``.
"""

import math

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont.io import ElementType


def _active_pairs(types: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pair0 = (types == ElementType.QUAD_TO.value) | (types == ElementType.CURVE_TO.value)
    pair1 = types == ElementType.CURVE_TO.value
    pair2 = (
        (types == ElementType.MOVE_TO.value)
        | (types == ElementType.LINE_TO.value)
        | (types == ElementType.QUAD_TO.value)
        | (types == ElementType.CURVE_TO.value)
    )
    return pair0, pair1, pair2


def _bbox_center(types: Tensor, coords: Tensor) -> Tensor:
    """Return the tight bounding-box centre via the Rust ``tight_bbox`` implementation.

    Delegates to :func:`torchfont._torchfont.tight_bbox`, which evaluates true
    curve extrema for QUAD_TO and CURVE_TO segments rather than bounding the
    control-point hull.
    """
    result = _torchfont.tight_bbox(
        types.cpu().contiguous().numpy(),
        coords.cpu().contiguous().reshape(-1).numpy(),
    )
    if result is None:
        return coords.new_zeros(2)
    x_min, y_min, x_max, y_max = result
    return coords.new_tensor([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])


def _apply_matrix(
    types: Tensor,
    coords: Tensor,
    matrix: Tensor,
    center: Tensor,
    translate: tuple[float, float],
) -> Tensor:
    """Apply ``p' = (p - center) @ matrix.T + center + translate`` to active pairs."""
    c = center
    t = coords.new_tensor(translate)
    active = torch.stack(list(_active_pairs(types)), dim=1).unsqueeze(-1)
    pts = coords.reshape(-1, 3, 2)
    transformed = (pts - c) @ matrix.T + c + t
    return torch.where(active, transformed, pts).reshape_as(coords)


def _rotation_scale_shear_matrix(
    angle_deg: float,
    scale: float,
    shear_deg: float,
    *,
    like: Tensor,
) -> Tensor:
    """Return a 2x2 matrix for scale * x-shear * rotation (all applied in place)."""
    a = math.radians(angle_deg)
    s = math.radians(shear_deg)
    cos_a, sin_a, tan_s = math.cos(a), math.sin(a), math.tan(s)
    return like.new_tensor(
        [
            [scale * (cos_a + sin_a * tan_s), scale * (-sin_a + cos_a * tan_s)],
            [scale * sin_a, scale * cos_a],
        ],
    )


def _preserve_closed_subpath_winding(
    types: Tensor,
    coords: Tensor,
) -> tuple[Tensor, Tensor]:
    out_types, out_coords = _torchfont.reverse_closed_subpaths(
        types.cpu().contiguous().numpy(),
        coords.cpu().contiguous().reshape(-1).numpy(),
    )
    return (
        torch.from_numpy(out_types).to(device=types.device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords.device),
    )


def horizontal_flip(
    types: Tensor,
    coords: Tensor,
    *,
    preserve_winding: bool = True,
) -> tuple[Tensor, Tensor]:
    """Flip a glyph outline horizontally around the bounding-box centre.

    Both on-curve endpoints and off-curve control points are transformed.
    Zero-coordinate element types (CLOSE, END, PAD) are left unchanged.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        preserve_winding: Reverse closed subpaths after reflection so their
            winding direction matches the input. Default: ``True``.

    Returns:
        A new ``(types, coords)`` pair with coordinates reflected around the
        bounding-box centre. Closed subpaths are re-encoded when
        ``preserve_winding`` is enabled.

    """
    matrix = coords.new_tensor([[-1.0, 0.0], [0.0, 1.0]])
    center = _bbox_center(types, coords)
    out_coords = _apply_matrix(types, coords, matrix, center, (0.0, 0.0))
    if preserve_winding:
        return _preserve_closed_subpath_winding(types, out_coords)
    return types, out_coords


def vertical_flip(
    types: Tensor,
    coords: Tensor,
    *,
    preserve_winding: bool = True,
) -> tuple[Tensor, Tensor]:
    """Flip a glyph outline vertically around the bounding-box centre.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        preserve_winding: Reverse closed subpaths after reflection so their
            winding direction matches the input. Default: ``True``.

    Returns:
        A new ``(types, coords)`` pair with coordinates reflected around the
        bounding-box centre. Closed subpaths are re-encoded when
        ``preserve_winding`` is enabled.

    """
    matrix = coords.new_tensor([[1.0, 0.0], [0.0, -1.0]])
    center = _bbox_center(types, coords)
    out_coords = _apply_matrix(types, coords, matrix, center, (0.0, 0.0))
    if preserve_winding:
        return _preserve_closed_subpath_winding(types, out_coords)
    return types, out_coords


def affine(
    types: Tensor,
    coords: Tensor,
    *,
    angle: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    shear: float = 0.0,
) -> tuple[Tensor, Tensor]:
    """Apply a deterministic affine transformation to a glyph outline.

    The transform composes **uniform scale**, **x-shear**, and **rotation**
    around the bounding-box centre, then applies ``translate``. Control points
    and endpoints are all transformed consistently; zero-coordinate element
    types (CLOSE, END, PAD) are not modified.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        angle: Counter-clockwise rotation in degrees.
        translate: Translation ``(tx, ty)`` in em units applied
            after rotation and scaling. Values must be finite.
        scale: Uniform scale factor (must be positive and finite).
        shear: x-shear angle in degrees.

    Returns:
        A new ``(types, coords)`` pair with the affine transform applied.
        ``types`` is returned unchanged (same object).

    """
    if not math.isfinite(scale) or scale <= 0:
        msg = "scale must be positive and finite"
        raise ValueError(msg)
    if math.isnan(angle):
        msg = "angle must be finite"
        raise ValueError(msg)
    if math.isnan(shear):
        msg = "shear must be finite"
        raise ValueError(msg)
    if not all(math.isfinite(value) for value in translate):
        msg = "translate values must be finite"
        raise ValueError(msg)
    matrix = _rotation_scale_shear_matrix(angle, scale, shear, like=coords)
    center = _bbox_center(types, coords)
    return types, _apply_matrix(types, coords, matrix, center, translate)
