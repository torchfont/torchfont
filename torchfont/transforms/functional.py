"""Deterministic functional kernels for semantic font data."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar, cast, overload

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont.datasets import GlyphRef, VariableGlyphRef
from torchfont.io import COORD_DIM, ElementType
from torchfont.transforms import bitmap as _bitmap
from torchfont.transforms import curves as _curves
from torchfont.transforms import geometric as _geometric
from torchfont.transforms import outline as _outline
from torchfont.transforms import subpath as _subpath
from torchfont.transforms._transform import Bitmap, Outline, OutlinePatches

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    import numpy as np
    from numpy.typing import NDArray

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")
U = TypeVar("U")

_KERNEL_REGISTRY: dict[object, dict[type[object], object]] = {}


def _dispatchable(
    input_type: type[S],
) -> Callable[[Callable[Concatenate[S, P], R]], Callable[Concatenate[S, P], R]]:
    def decorator(
        kernel: Callable[Concatenate[S, P], R],
    ) -> Callable[Concatenate[S, P], R]:
        @functools.wraps(kernel)
        def dispatcher(inpt: S, *args: P.args, **kwargs: P.kwargs) -> R:
            registered = _KERNEL_REGISTRY[dispatcher]
            for cls in type(inpt).__mro__:
                if candidate := registered.get(cls):
                    resolved = cast("Callable[Concatenate[S, P], R]", candidate)
                    return resolved(inpt, *args, **kwargs)
            msg = f"{dispatcher.__name__} does not support {type(inpt).__name__}"
            raise TypeError(msg)

        _KERNEL_REGISTRY[dispatcher] = {input_type: kernel}
        return dispatcher

    return decorator


def register_kernel(
    functional: Callable[Concatenate[S, P], R], input_type: type[U]
) -> Callable[[Callable[Concatenate[U, P], R]], Callable[Concatenate[U, P], R]]:
    """Register a type-specific kernel for a TorchFont functional."""
    if functional not in _KERNEL_REGISTRY:
        msg = "kernels can only be registered for TorchFont functionals"
        raise ValueError(msg)

    def decorator(
        kernel: Callable[Concatenate[U, P], R],
    ) -> Callable[Concatenate[U, P], R]:
        registry = _KERNEL_REGISTRY[functional]
        if input_type in registry:
            msg = f"a kernel is already registered for {input_type.__name__}"
            raise ValueError(msg)
        registry[input_type] = kernel
        return kernel

    return decorator


def _native_outline(
    inpt: Outline,
    operation: Callable[
        ..., tuple[NDArray[np.integer[Any]], NDArray[np.floating[Any]]]
    ],
    *args: object,
) -> Outline:
    types_device = inpt.types.device
    coords_device = inpt.coords.device
    types = inpt.types.cpu().contiguous()
    coords = inpt.coords.cpu().contiguous()
    out_types, out_coords = operation(types.numpy(), coords.reshape(-1).numpy(), *args)
    return Outline(
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, COORD_DIM).to(device=coords_device),
    )


@overload
def load_glyph(ref: GlyphRef) -> Outline: ...


@overload
def load_glyph(
    ref: VariableGlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline: ...


def load_glyph(
    ref: GlyphRef | VariableGlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline:
    """Load one glyph outline at an explicit, deterministic location."""
    if isinstance(ref, GlyphRef):
        if location is not None:
            msg = "location cannot override a GlyphRef location"
            raise ValueError(msg)
        location = ref.location
    raw_types, raw_coords = _torchfont.load_glyph(
        ref.font.path,
        ref.font.ttc_index,
        ref.codepoint,
        None
        if location is None
        else {str(tag): float(value) for tag, value in location.items()},
    )
    return Outline(
        torch.from_numpy(raw_types),
        torch.from_numpy(raw_coords).view(-1, COORD_DIM),
    )


@_dispatchable(Outline)
def quad_to_cubic(inpt: Outline, *, merge_curves: bool = False) -> Outline:
    """Convert quadratic segments to cubic segments."""
    return Outline(
        *_curves.quad_to_cubic(inpt.types, inpt.coords, merge_curves=merge_curves)
    )


@_dispatchable(Outline)
def cubic_to_quad(inpt: Outline) -> Outline:
    """Convert cubic segments to quadratic segments."""
    return Outline(*_curves.cubic_to_quad(inpt.types, inpt.coords))


@_dispatchable(Outline)
def merge_curves(inpt: Outline) -> Outline:
    """Merge adjacent pieces of the same parent curve or line."""
    return Outline(*_curves.merge_curves(inpt.types, inpt.coords))


@_dispatchable(Outline)
def remove_overlaps(inpt: Outline) -> Outline:
    """Merge overlapping subpaths."""
    return Outline(*_outline.remove_overlaps(inpt.types, inpt.coords))


@_dispatchable(Outline)
def normalize_subpath_start_points(inpt: Outline) -> Outline:
    """Choose a deterministic start point for each closed subpath."""
    return Outline(*_subpath.normalize_subpath_start_points(inpt.types, inpt.coords))


@_dispatchable(Outline)
def horizontal_flip(inpt: Outline, *, preserve_winding: bool = True) -> Outline:
    """Flip an outline horizontally around its bounding-box centre."""
    return Outline(
        *_geometric.horizontal_flip(
            inpt.types, inpt.coords, preserve_winding=preserve_winding
        )
    )


@_dispatchable(Outline)
def vertical_flip(inpt: Outline, *, preserve_winding: bool = True) -> Outline:
    """Flip an outline vertically around its bounding-box centre."""
    return Outline(
        *_geometric.vertical_flip(
            inpt.types, inpt.coords, preserve_winding=preserve_winding
        )
    )


@_dispatchable(Outline)
def affine(
    inpt: Outline,
    *,
    angle: float = 0.0,
    translate: tuple[float, float] = (0.0, 0.0),
    scale: float = 1.0,
    shear: float = 0.0,
) -> Outline:
    """Apply a deterministic affine transformation."""
    return Outline(
        *_geometric.affine(
            inpt.types,
            inpt.coords,
            angle=angle,
            translate=translate,
            scale=scale,
            shear=shear,
        )
    )


@_dispatchable(Outline)
def coord_jitter(inpt: Outline, noise: Tensor) -> Outline:
    """Add caller-provided noise to active coordinate pairs."""
    types, coords = inpt.types, inpt.coords
    pair0 = (types == ElementType.QUAD_TO.value) | (types == ElementType.CURVE_TO.value)
    pair1 = types == ElementType.CURVE_TO.value
    pair2 = (
        (types == ElementType.MOVE_TO.value)
        | (types == ElementType.LINE_TO.value)
        | (types == ElementType.QUAD_TO.value)
        | (types == ElementType.CURVE_TO.value)
    )
    active = torch.stack((pair0, pair1, pair2), dim=1).unsqueeze(-1)
    points = coords.reshape(-1, 3, 2)
    noise = noise[: types.size(0)].to(device=coords.device, dtype=coords.dtype)
    return Outline(
        types, torch.where(active, points + noise, points).reshape_as(coords)
    )


@_dispatchable(Outline)
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


@_dispatchable(Outline)
def remove_overlap_groups(inpt: Outline, selection_values: Tensor) -> Outline:
    """Simplify overlap groups according to explicit selection values."""
    return _native_outline(
        inpt,
        _torchfont.random_remove_overlaps,
        selection_values.cpu().contiguous().numpy(),
    )


@_dispatchable(Outline)
def set_subpath_start_points(inpt: Outline, selection_values: Tensor) -> Outline:
    """Set closed-subpath start points from explicit unit-interval values."""
    return _native_outline(
        inpt,
        _torchfont.randomize_subpath_start_points,
        selection_values.cpu().contiguous().numpy(),
    )


@_dispatchable(Outline)
def reorder_subpaths(inpt: Outline, keys: Tensor) -> Outline:
    """Order subpaths by explicit sort keys."""
    return _native_outline(
        inpt,
        _torchfont.randomize_subpath_order,
        keys.cpu().contiguous().numpy(),
    )


@_dispatchable(Outline)
def patchify(inpt: Outline, patch_size: int) -> OutlinePatches:
    """Split an outline into fixed-length patches."""
    return OutlinePatches(*_outline.patchify(inpt.types, inpt.coords, patch_size))


@_dispatchable(Outline)
def render_bitmap(
    inpt: Outline,
    size: int = 64,
    mode: _bitmap.BitmapMode = "bbox_square",
    fill_rule: _bitmap.FillRule = "winding",
) -> Bitmap:
    """Render an outline into a greyscale bitmap."""
    return Bitmap(_bitmap.render_bitmap(inpt.types, inpt.coords, size, mode, fill_rule))


__all__ = [
    "affine",
    "coord_jitter",
    "cubic_to_quad",
    "horizontal_flip",
    "load_glyph",
    "merge_curves",
    "normalize_subpath_start_points",
    "patchify",
    "quad_to_cubic",
    "register_kernel",
    "remove_overlap_groups",
    "remove_overlaps",
    "render_bitmap",
    "reorder_subpaths",
    "set_subpath_start_points",
    "split_segments",
    "vertical_flip",
]
