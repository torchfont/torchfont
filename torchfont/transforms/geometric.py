"""Geometric transformation functions for glyph outline tensors.

All functions follow the same convention as :mod:`torchfont.transforms`:
they accept ``(types, coords)`` and return a transformed ``(types, coords)``
pair without modifying the inputs.

Coordinate layout (``coords`` shape ``(N, 6)``)::

    [cx0, cy0, cx1, cy1, x, y]

    Pair 0 (cx0, cy0): off-curve control point 1 — active for QUAD_TO / CURVE_TO
    Pair 1 (cx1, cy1): off-curve control point 2 — active for CURVE_TO only
    Pair 2 (x,   y  ): on-curve endpoint        — active for all drawing commands

All coordinates are UPM-normalised. The glyph body typically occupies
``[0, 1] x [0, 1]`` inside the full canvas ``[-0.25, 1.25] x [-0.25, 1.25]``.
"""

import math

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont.io import CommandType


def _active_pairs(types: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    pair0 = (types == CommandType.QUAD_TO.value) | (types == CommandType.CURVE_TO.value)
    pair1 = types == CommandType.CURVE_TO.value
    pair2 = (
        (types == CommandType.MOVE_TO.value)
        | (types == CommandType.LINE_TO.value)
        | (types == CommandType.QUAD_TO.value)
        | (types == CommandType.CURVE_TO.value)
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
) -> Tensor:
    """Return a 2x2 matrix for scale * x-shear * rotation (all applied in place)."""
    a = math.radians(angle_deg)
    s = math.radians(shear_deg)
    cos_a, sin_a, tan_s = math.cos(a), math.sin(a), math.tan(s)
    return torch.tensor(
        [
            [scale * (cos_a + sin_a * tan_s), scale * (-sin_a + cos_a * tan_s)],
            [scale * sin_a, scale * cos_a],
        ],
        dtype=torch.float32,
    )


def horizontal_flip(
    types: Tensor,
    coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Flip a glyph outline horizontally around the bounding-box centre.

    Both on-curve endpoints and off-curve control points are transformed.
    Zero-padded entries (CLOSE, END, PAD) are left unchanged.

    Note:
        Flipping reverses contour winding order. For most sequence models
        this is acceptable; take care when consistent winding is required.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new ``(types, coords)`` pair with coordinates reflected around the
        bounding-box centre. ``types`` is returned unchanged (same object).

    """
    matrix = torch.tensor([[-1.0, 0.0], [0.0, 1.0]])
    center = _bbox_center(types, coords)
    return types, _apply_matrix(types, coords, matrix, center, (0.0, 0.0))


def vertical_flip(
    types: Tensor,
    coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Flip a glyph outline vertically around the bounding-box centre.

    Note:
        Flipping reverses contour winding order. For most sequence models
        this is acceptable; take care when consistent winding is required.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new ``(types, coords)`` pair with coordinates reflected around the
        bounding-box centre. ``types`` is returned unchanged (same object).

    """
    matrix = torch.tensor([[1.0, 0.0], [0.0, -1.0]])
    center = _bbox_center(types, coords)
    return types, _apply_matrix(types, coords, matrix, center, (0.0, 0.0))


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
    and endpoints are all transformed consistently; padding entries (CLOSE, END,
    PAD) are not modified.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        angle: Counter-clockwise rotation in degrees.
        translate: Translation ``(tx, ty)`` in UPM-normalised units applied
            after rotation and scaling.
        scale: Uniform scale factor (must be positive).
        shear: x-shear angle in degrees.

    Returns:
        A new ``(types, coords)`` pair with the affine transform applied.
        ``types`` is returned unchanged (same object).

    """
    if scale <= 0:
        msg = "scale must be positive"
        raise ValueError(msg)
    matrix = _rotation_scale_shear_matrix(angle, scale, shear)
    center = _bbox_center(types, coords)
    return types, _apply_matrix(types, coords, matrix, center, translate)


def random_horizontal_flip(
    types: Tensor,
    coords: Tensor,
    *,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomly flip a glyph outline horizontally with probability ``p``.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        p: Probability of applying the flip. Default: ``0.5``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Either the original ``(types, coords)`` unchanged, or a flipped copy.

    """
    if torch.rand(1, generator=generator).item() < p:
        return horizontal_flip(types, coords)
    return types, coords


def random_vertical_flip(
    types: Tensor,
    coords: Tensor,
    *,
    p: float = 0.5,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomly flip a glyph outline vertically with probability ``p``.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        p: Probability of applying the flip. Default: ``0.5``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Either the original ``(types, coords)`` unchanged, or a flipped copy.

    """
    if torch.rand(1, generator=generator).item() < p:
        return vertical_flip(types, coords)
    return types, coords


def _sym_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        return (-abs(float(value)), abs(float(value)))
    lo, hi = float(value[0]), float(value[1])
    if lo > hi:
        msg = "range must satisfy min <= max"
        raise ValueError(msg)
    return (lo, hi)


def _validate_scale_range(scale: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(scale[0]), float(scale[1])
    if lo <= 0 or hi <= 0:
        msg = "scale values must be positive"
        raise ValueError(msg)
    if lo > hi:
        msg = "scale range must satisfy min <= max"
        raise ValueError(msg)
    return (lo, hi)


def random_affine(
    types: Tensor,
    coords: Tensor,
    *,
    degrees: float | tuple[float, float] = 0.0,
    translate: tuple[float, float] | None = None,
    scale: tuple[float, float] | None = None,
    shear: float | tuple[float, float] = 0.0,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Apply a random affine transformation to a glyph outline.

    Each parameter is sampled independently and uniformly from its range.
    The transform composes **uniform scale**, **x-shear**, **rotation**, and
    **translation** around the bounding-box centre. Control points and endpoints
    are all transformed consistently; padding entries (CLOSE, END, PAD) are not
    modified.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        degrees: Rotation range in degrees. A single float ``d`` gives
            ``[-d, d]``; a ``(min, max)`` tuple is used directly.
        translate: Maximum absolute translation ``(max_dx, max_dy)`` in
            UPM-normalised units. Each axis is sampled uniformly from
            ``[-max_d, max_d]``. Default: no translation.
        scale: Scale range ``(min, max)``. Values must be positive and
            satisfy ``min <= max``. Default: no scaling.
        shear: x-shear range in degrees. Same format as ``degrees``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A new ``(types, coords)`` pair with the sampled affine applied.
        ``types`` is returned unchanged (same object).

    """
    deg_lo, deg_hi = _sym_range(degrees)
    shear_lo, shear_hi = _sym_range(shear)

    r = torch.rand(5, generator=generator)

    def _u(lo: float, hi: float, i: int) -> float:
        return lo + (hi - lo) * r[i].item()

    angle = _u(deg_lo, deg_hi, 0)
    tx = _u(-translate[0], translate[0], 1) if translate is not None else 0.0
    ty = _u(-translate[1], translate[1], 2) if translate is not None else 0.0
    sc = _u(*_validate_scale_range(scale), 3) if scale is not None else 1.0
    sh = _u(shear_lo, shear_hi, 4)

    return affine(
        types,
        coords,
        angle=angle,
        translate=(tx, ty),
        scale=sc,
        shear=sh,
    )


def random_coord_jitter(
    types: Tensor,
    coords: Tensor,
    *,
    std: float,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Add independent Gaussian noise to each active outline coordinate.

    Noise is sampled per scalar coordinate value with standard deviation
    ``std`` in UPM-normalised units. Non-active padding entries (CLOSE, END,
    PAD) and unused zero-padding columns (e.g. ``cx1, cy1`` for QUAD_TO) are
    not perturbed.

    A value of ``std=0.005`` corresponds to roughly 5 font-units in a
    1000-UPM font — a subtle perturbation that rarely changes the perceived
    glyph shape.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        std: Standard deviation of the Gaussian noise in UPM-normalised units.
            Must be non-negative.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A new ``(types, coords)`` pair with noise added to active coordinates.
        ``types`` is returned unchanged (same object).

    """
    if std < 0:
        msg = "std must be non-negative"
        raise ValueError(msg)
    if std == 0.0:
        return types, coords
    active = torch.stack(list(_active_pairs(types)), dim=1).unsqueeze(-1)
    pts = coords.reshape(-1, 3, 2)
    noise = torch.empty_like(pts).normal_(std=std, generator=generator)
    return types, torch.where(active, pts + noise, pts).reshape_as(coords)
