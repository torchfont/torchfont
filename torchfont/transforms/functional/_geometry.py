"""Functional geometric kernels for glyph outlines.

Public functions accept and return :class:`torchfont.Outline`
objects without modifying the input. Private tensor helpers operate on the
underlying ``(types, coords)`` pair.

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

from torchfont import _ops
from torchfont._outline import ElementType, Outline
from torchfont.transforms.functional._utils import (
    _require_no_grad,
    _same_types,
)


def _is_nan(value: float) -> bool:
    """Return whether ``value`` is NaN, without calling :func:`math.isnan`.

    Dynamo treats ``math.isnan`` as an operator returning a non-Tensor and cannot
    trace it once ``torch.compile`` runs with ``dynamic=True``, which makes these
    float parameters symbolic. NaN is the only float that compares false against
    both zero comparisons.
    """
    return not (value <= 0.0 or value > 0.0)


def _is_infinite(value: float) -> bool:
    """Return whether ``value`` is infinite, without calling :func:`math.isfinite`.

    Traceable for the same reason as :func:`_is_nan`.
    """
    return abs(value) == math.inf


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
    """Return the tight bounding-box centre as a ``(2,)`` tensor.

    Delegates to the ``torchfont::bbox_center`` operator, which evaluates true
    curve extrema for QUAD_TO and CURVE_TO segments rather than bounding the
    control-point hull.

    The centre is the reference frame a transform is applied around, not a
    differentiable output, so it is computed from detached coordinates. Gradients
    therefore flow through the transformed coordinates but not through the choice
    of centre.
    """
    return _ops.bbox_center(types.detach(), coords.detach())


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
    return _ops.reverse_closed_subpaths(types.detach(), coords.detach())


def _horizontal_flip(
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
        coords: 2-D floating point tensor of shape ``(N, 6)``.
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


def _vertical_flip(
    types: Tensor,
    coords: Tensor,
    *,
    preserve_winding: bool = True,
) -> tuple[Tensor, Tensor]:
    """Flip a glyph outline vertically around the bounding-box centre.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D floating point tensor of shape ``(N, 6)``.
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


def _affine(
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
        coords: 2-D floating point tensor of shape ``(N, 6)``.
        angle: Counter-clockwise rotation in degrees.
        translate: Translation ``(tx, ty)`` in em units applied
            after rotation and scaling. Values must be finite.
        scale: Uniform scale factor (must be positive and finite).
        shear: x-shear angle in degrees.

    Returns:
        A new ``(types, coords)`` pair with the affine transform applied.
        ``types`` is returned unchanged (same object).

    """
    if _is_nan(scale) or _is_infinite(scale) or scale <= 0:
        msg = "scale must be positive and finite"
        raise ValueError(msg)
    if _is_nan(angle):
        msg = "angle must be finite"
        raise ValueError(msg)
    if _is_nan(shear):
        msg = "shear must be finite"
        raise ValueError(msg)
    if any(_is_nan(value) or _is_infinite(value) for value in translate):
        msg = "translate values must be finite"
        raise ValueError(msg)
    matrix = _rotation_scale_shear_matrix(angle, scale, shear, like=coords)
    center = _bbox_center(types, coords)
    return types, _apply_matrix(types, coords, matrix, center, translate)


def horizontal_flip(inpt: Outline, *, preserve_winding: bool = True) -> Outline:
    """Flip an outline horizontally around its tight bounding-box centre.

    Differentiable only when ``preserve_winding`` is ``False``; reversing
    subpaths reorders elements in Rust and defines no gradient.
    """
    if preserve_winding:
        _require_no_grad(inpt, "horizontal_flip(preserve_winding=True)")
    out_types, out_coords = _horizontal_flip(
        inpt.types, inpt.coords, preserve_winding=preserve_winding
    )
    return Outline._wrap(out_types, out_coords)  # noqa: SLF001


def vertical_flip(inpt: Outline, *, preserve_winding: bool = True) -> Outline:
    """Flip an outline vertically around its tight bounding-box centre.

    Differentiable only when ``preserve_winding`` is ``False``; reversing
    subpaths reorders elements in Rust and defines no gradient.
    """
    if preserve_winding:
        _require_no_grad(inpt, "vertical_flip(preserve_winding=True)")
    out_types, out_coords = _vertical_flip(
        inpt.types, inpt.coords, preserve_winding=preserve_winding
    )
    return Outline._wrap(out_types, out_coords)  # noqa: SLF001


def affine(
    inpt: Outline,
    *,
    angle: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    shear: float = 0.0,
) -> Outline:
    """Apply a deterministic affine transformation.

    Differentiable with respect to ``coords``. The bounding-box centre the
    transform pivots around is treated as a constant reference frame.
    """
    return _same_types(
        inpt,
        _affine(
            inpt.types,
            inpt.coords,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
        )[1],
    )


def add_coordinate_noise(inpt: Outline, noise: Tensor) -> Outline:
    """Add caller-provided noise to active coordinate pairs.

    Differentiable with respect to both ``coords`` and ``noise``.
    """
    types, coords = inpt.types, inpt.coords
    noise_tail_shape = (3, 2)
    if (
        noise.ndim != len(noise_tail_shape) + 1
        or tuple(noise.shape[1:]) != noise_tail_shape
    ):
        msg = f"noise must have shape (N, 3, 2), got {tuple(noise.shape)}"
        raise ValueError(msg)
    if noise.shape[0] < types.shape[0]:
        msg = "noise must have at least as many rows as the outline"
        raise ValueError(msg)
    active = torch.stack(_active_pairs(types), dim=1).unsqueeze(-1)
    points = coords.reshape(-1, 3, 2)
    noise = noise[: types.size(0)].to(device=coords.device, dtype=coords.dtype)
    return _same_types(
        inpt, torch.where(active, points + noise, points).reshape_as(coords)
    )


__all__ = ["add_coordinate_noise", "affine", "horizontal_flip", "vertical_flip"]
