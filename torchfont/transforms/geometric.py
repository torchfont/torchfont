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


def _random_device(generator: torch.Generator | None, fallback: Tensor) -> torch.device:
    return generator.device if generator is not None else fallback.device


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


def random_horizontal_flip(
    types: Tensor,
    coords: Tensor,
    *,
    p: float = 0.5,
    preserve_winding: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomly flip a glyph outline horizontally with probability ``p``.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        p: Probability of applying the flip. Default: ``0.5``.
        preserve_winding: Reverse closed subpaths after reflection so their
            winding direction matches the input. Default: ``True``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Either the original ``(types, coords)`` unchanged, or a flipped copy.

    """
    if (
        torch.rand(
            1,
            device=_random_device(generator, coords),
            generator=generator,
        ).item()
        < p
    ):
        return horizontal_flip(types, coords, preserve_winding=preserve_winding)
    return types, coords


def random_vertical_flip(
    types: Tensor,
    coords: Tensor,
    *,
    p: float = 0.5,
    preserve_winding: bool = True,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomly flip a glyph outline vertically with probability ``p``.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        p: Probability of applying the flip. Default: ``0.5``.
        preserve_winding: Reverse closed subpaths after reflection so their
            winding direction matches the input. Default: ``True``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        Either the original ``(types, coords)`` unchanged, or a flipped copy.

    """
    if (
        torch.rand(
            1,
            device=_random_device(generator, coords),
            generator=generator,
        ).item()
        < p
    ):
        return vertical_flip(types, coords, preserve_winding=preserve_winding)
    return types, coords


def _sym_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (int, float)):
        limit = float(value)
        if not math.isfinite(limit):
            msg = "range values must be finite"
            raise ValueError(msg)
        return (-abs(limit), abs(limit))
    lo, hi = float(value[0]), float(value[1])
    if not math.isfinite(lo) or not math.isfinite(hi):
        msg = "range values must be finite"
        raise ValueError(msg)
    return (lo, hi)


def _validate_scale_range(scale: tuple[float, float]) -> tuple[float, float]:
    lo, hi = float(scale[0]), float(scale[1])
    if not math.isfinite(lo) or not math.isfinite(hi) or lo <= 0 or hi <= 0:
        msg = "scale values must be positive and finite"
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
    are all transformed consistently; zero-coordinate element types (CLOSE, END)
    and padding entries (PAD) are not modified.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        degrees: Rotation range in degrees. A single float ``d`` gives
            ``[-d, d]``; a ``(min, max)`` tuple is used directly.
        translate: Maximum absolute translation ``(max_dx, max_dy)`` in em
            units. Each axis is sampled uniformly from ``[-max_d, max_d]``.
            Values must be finite. Default: no translation.
        scale: Scale range ``(min, max)``. Values must be positive and finite.
            Default: no scaling.
        shear: x-shear range in degrees. Same format as ``degrees``.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A new ``(types, coords)`` pair with the sampled affine applied.
        ``types`` is returned unchanged (same object).

    """
    deg_lo, deg_hi = _sym_range(degrees)
    shear_lo, shear_hi = _sym_range(shear)

    r = torch.rand(
        5,
        device=_random_device(generator, coords),
        generator=generator,
    )

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
    """Add independent Gaussian noise to each active value in the outline coordinates.

    Noise is sampled per scalar value in ``coords`` with standard deviation
    ``std`` in em units. Non-active zero-coordinate element types
    (CLOSE, END, PAD) and unused zero-padding columns (e.g. ``cx1, cy1`` for
    QUAD_TO) are not perturbed.

    A value of ``std=0.005`` corresponds to roughly 5 font-units in a
    1000-UPM font — a subtle perturbation that rarely changes the perceived
    glyph shape.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        std: Standard deviation of the Gaussian noise in em units.
            Must be finite.
        generator: Optional ``torch.Generator`` for reproducible sampling.

    Returns:
        A new ``(types, coords)`` pair with noise added to active coordinates.
        ``types`` is returned unchanged (same object).

    """
    if not math.isfinite(std):
        msg = "std must be finite"
        raise ValueError(msg)
    if std == 0.0:
        return types, coords
    active = torch.stack(list(_active_pairs(types)), dim=1).unsqueeze(-1)
    pts = coords.reshape(-1, 3, 2)
    noise = torch.empty(
        pts.shape,
        dtype=pts.dtype,
        device=_random_device(generator, coords),
    ).normal_(std=std, generator=generator)
    noise = noise.to(device=coords.device)
    return types, torch.where(active, pts + noise, pts).reshape_as(coords)
