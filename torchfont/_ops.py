"""Rust outline kernels registered as PyTorch custom operators.

Each Rust kernel crosses a CPU and NumPy boundary. Calling one directly from a
compiled region would break the graph, because Dynamo cannot trace into an
extension module. Registering the boundary with :func:`torch.library.custom_op`
turns each kernel into a single opaque graph node instead, so
:func:`torch.compile` captures a whole outline pipeline without breaking.

Every operator here:

* takes and returns tensors, never NumPy arrays or Python scalars, so no value
  escapes into the graph as a constant;
* is registered only for CPU tensors, matching the device of the Rust kernel;
* accepts the native kernel's actual dtypes: ``torch.long`` element types and
  ``torch.float32`` coordinates;
* declares a fake implementation so shape propagation works without running the
  kernel. Most of these kernels change the number of path elements, which makes
  the output length data-dependent; those fakes allocate an unbacked dynamic
  size.

None of them register an autograd formula. They reorder or re-encode path
elements, so no gradient is defined. Callers reject outlines that require grad
before reaching this layer, which produces an error naming the kernel.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont._outline import COORD_DIM

if TYPE_CHECKING:
    import numpy as np

    from torchfont._torchfont import _BitmapMode, _FillRule


def _arrays(types: Tensor, coords: Tensor) -> tuple[np.ndarray, np.ndarray]:
    """Return NumPy views accepted by the CPU float32 Rust kernels."""
    if types.dtype is not torch.long:
        msg = f"types must have dtype torch.long, got {types.dtype}"
        raise TypeError(msg)
    if coords.dtype is not torch.float32:
        msg = f"coords must have dtype torch.float32, got {coords.dtype}"
        raise TypeError(msg)
    return (
        types.detach().contiguous().numpy(),
        coords.detach().contiguous().reshape(-1).numpy(),
    )


def _restore(
    out_types: np.ndarray,
    out_coords: np.ndarray,
) -> tuple[Tensor, Tensor]:
    """Rebuild CPU tensors returned by the native kernel."""
    return (
        torch.from_numpy(out_types),
        torch.from_numpy(out_coords).view(-1, COORD_DIM),
    )


def _dynamic_outline(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Allocate a fake outline whose element count is data-dependent."""
    length = torch.library.get_ctx().new_dynamic_size()
    return (
        types.new_empty(length),
        coords.new_empty(length, COORD_DIM),
    )


def _selection(values: Tensor) -> np.ndarray:
    if values.dtype is not torch.float32:
        msg = f"selection values must have dtype torch.float32, got {values.dtype}"
        raise TypeError(msg)
    return values.detach().contiguous().numpy()


@torch.library.custom_op(
    "torchfont::quad_to_cubic",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords, bool merge_curves) -> (Tensor, Tensor)",
)
def quad_to_cubic(
    types: Tensor, coords: Tensor, merge_curves: bool
) -> tuple[Tensor, Tensor]:
    """Convert quadratic segments to cubic segments."""
    out = _torchfont.quad_to_cubic(*_arrays(types, coords), merge_curves)
    return _restore(*out)


@quad_to_cubic.register_fake
def _(types: Tensor, coords: Tensor, merge_curves: bool) -> tuple[Tensor, Tensor]:
    if merge_curves:
        return _dynamic_outline(types, coords)
    return types.new_empty(types.shape), coords.new_empty(coords.shape)


@torch.library.custom_op(
    "torchfont::cubic_to_quad",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords) -> (Tensor, Tensor)",
)
def cubic_to_quad(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Convert cubic segments to sequences of quadratic segments."""
    out = _torchfont.cubic_to_quad(*_arrays(types, coords))
    return _restore(*out)


@cubic_to_quad.register_fake
def _(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::merge_curves",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords) -> (Tensor, Tensor)",
)
def merge_curves(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Merge adjacent pieces of the same parent curve or line."""
    out = _torchfont.merge_curves(*_arrays(types, coords))
    return _restore(*out)


@merge_curves.register_fake
def _(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::split_segments",
    mutates_args=(),
    device_types="cpu",
    schema=(
        "(Tensor types, Tensor coords, Tensor selection_values, "
        "Tensor position_values, float split_probability, float[] split_range) "
        "-> (Tensor, Tensor)"
    ),
)
def split_segments(
    types: Tensor,
    coords: Tensor,
    selection_values: Tensor,
    position_values: Tensor,
    split_probability: float,
    split_range: list[float],
) -> tuple[Tensor, Tensor]:
    """Split segments according to explicit selection and position values.

    ``split_range`` is a list rather than a tuple because operator schemas do not
    support tuple arguments.
    """
    low, high = split_range
    out = _torchfont.random_split_segments(
        *_arrays(types, coords),
        _selection(selection_values),
        _selection(position_values),
        split_probability,
        (low, high),
    )
    return _restore(*out)


@split_segments.register_fake
def _(
    types: Tensor,
    coords: Tensor,
    selection_values: Tensor,
    position_values: Tensor,
    split_probability: float,
    split_range: list[float],
) -> tuple[Tensor, Tensor]:
    del selection_values, position_values, split_probability, split_range
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::remove_overlaps",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords) -> (Tensor, Tensor)",
)
def remove_overlaps(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Merge overlapping subpaths with Skia PathOps winding simplification."""
    out = _torchfont.remove_overlaps(*_arrays(types, coords))
    return _restore(*out)


@remove_overlaps.register_fake
def _(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::remove_overlap_groups",
    mutates_args=(),
    device_types="cpu",
    schema=(
        "(Tensor types, Tensor coords, Tensor selection_values) -> (Tensor, Tensor)"
    ),
)
def remove_overlap_groups(
    types: Tensor, coords: Tensor, selection_values: Tensor
) -> tuple[Tensor, Tensor]:
    """Simplify overlap groups according to explicit selection values."""
    out = _torchfont.random_remove_overlaps(
        *_arrays(types, coords), _selection(selection_values)
    )
    return _restore(*out)


@remove_overlap_groups.register_fake
def _(types: Tensor, coords: Tensor, selection_values: Tensor) -> tuple[Tensor, Tensor]:
    del selection_values
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::normalize_subpath_start_points",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords) -> (Tensor, Tensor)",
)
def normalize_subpath_start_points(
    types: Tensor, coords: Tensor
) -> tuple[Tensor, Tensor]:
    """Choose a deterministic start point for each closed subpath."""
    out = _torchfont.normalize_subpath_start_points(*_arrays(types, coords))
    return _restore(*out)


@normalize_subpath_start_points.register_fake
def _(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::set_subpath_start_points",
    mutates_args=(),
    device_types="cpu",
    schema=(
        "(Tensor types, Tensor coords, Tensor selection_values) -> (Tensor, Tensor)"
    ),
)
def set_subpath_start_points(
    types: Tensor, coords: Tensor, selection_values: Tensor
) -> tuple[Tensor, Tensor]:
    """Set closed-subpath start points from explicit unit-interval values."""
    out = _torchfont.randomize_subpath_start_points(
        *_arrays(types, coords), _selection(selection_values)
    )
    return _restore(*out)


@set_subpath_start_points.register_fake
def _(types: Tensor, coords: Tensor, selection_values: Tensor) -> tuple[Tensor, Tensor]:
    del selection_values
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::reorder_subpaths",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords, Tensor keys) -> (Tensor, Tensor)",
)
def reorder_subpaths(
    types: Tensor, coords: Tensor, keys: Tensor
) -> tuple[Tensor, Tensor]:
    """Order subpaths by explicit sort keys."""
    out = _torchfont.randomize_subpath_order(*_arrays(types, coords), _selection(keys))
    return _restore(*out)


@reorder_subpaths.register_fake
def _(types: Tensor, coords: Tensor, keys: Tensor) -> tuple[Tensor, Tensor]:
    del keys
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::reverse_closed_subpaths",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords) -> (Tensor, Tensor)",
)
def reverse_closed_subpaths(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Reverse the winding direction of every closed subpath."""
    out = _torchfont.reverse_closed_subpaths(*_arrays(types, coords))
    return _restore(*out)


@reverse_closed_subpaths.register_fake
def _(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    return _dynamic_outline(types, coords)


@torch.library.custom_op(
    "torchfont::bbox_center",
    mutates_args=(),
    device_types="cpu",
    schema="(Tensor types, Tensor coords) -> Tensor",
)
def bbox_center(types: Tensor, coords: Tensor) -> Tensor:
    """Return the tight bounding-box centre as a ``(2,)`` tensor.

    Empty outlines have no bounding box and yield the origin, matching the
    reference frame an affine transform would use for them.
    """
    result = _torchfont.tight_bbox(*_arrays(types, coords))
    if result is None:
        return coords.new_zeros(2)
    x_min, y_min, x_max, y_max = result
    return coords.new_tensor([(x_min + x_max) / 2.0, (y_min + y_max) / 2.0])


@bbox_center.register_fake
def _(types: Tensor, coords: Tensor) -> Tensor:
    del types
    return coords.new_empty(2)


@torch.library.custom_op(
    "torchfont::render_bitmap",
    mutates_args=(),
    device_types="cpu",
    schema=(
        "(Tensor types, Tensor coords, int size, str mode, str fill_rule, "
        "bool antialias) -> Tensor"
    ),
)
def render_bitmap(
    types: Tensor,
    coords: Tensor,
    size: int,
    mode: str,
    fill_rule: str,
    antialias: bool,
) -> Tensor:
    """Rasterize an outline into a ``uint8`` greyscale ``H x W`` tensor.

    ``mode`` and ``fill_rule`` are plain strings because operator schemas have no
    literal string type. The Rust kernel rejects an unknown value, so they are
    passed through rather than validated again here.
    """
    raw, width, height = _torchfont.render_bitmap(
        *_arrays(types, coords),
        size,
        cast("_BitmapMode", mode),
        cast("_FillRule", fill_rule),
        antialias,
    )
    return torch.from_numpy(raw).view(height, width)


@render_bitmap.register_fake
def _(
    types: Tensor,
    coords: Tensor,
    size: int,
    mode: str,
    fill_rule: str,
    antialias: bool,
) -> Tensor:
    del types, fill_rule, antialias
    options = torch.empty(0, dtype=torch.uint8, device=coords.device)
    if mode == "bbox":
        ctx = torch.library.get_ctx()
        return options.new_empty((ctx.new_dynamic_size(), ctx.new_dynamic_size()))
    return options.new_empty((size, size))


__all__ = [
    "bbox_center",
    "cubic_to_quad",
    "merge_curves",
    "normalize_subpath_start_points",
    "quad_to_cubic",
    "remove_overlap_groups",
    "remove_overlaps",
    "render_bitmap",
    "reorder_subpaths",
    "reverse_closed_subpaths",
    "set_subpath_start_points",
    "split_segments",
]
