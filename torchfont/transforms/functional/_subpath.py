"""Functional subpath kernels."""

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont.structures import Outline
from torchfont.transforms.functional._utils import _native_outline


def _normalize_subpath_start_points(
    types: Tensor,
    coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Move each subpath start to its lexicographically smallest endpoint.

    ``(x, y)`` endpoint order is used as the deterministic key. Open subpaths
    (those without a closing ``Close``), ``END``, and ``PAD`` element types are
    returned unchanged. When rotation crosses the old closing edge, that implicit
    edge is materialised as ``LINE_TO`` so the represented geometry is preserved.
    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.normalize_subpath_start_points(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )


def normalize_subpath_start_points(inpt: Outline) -> Outline:
    """Choose a deterministic start point for each closed subpath."""
    return Outline(*_normalize_subpath_start_points(inpt.types, inpt.coords))


def set_subpath_start_points(inpt: Outline, selection_values: Tensor) -> Outline:
    """Set closed-subpath start points from explicit unit-interval values."""
    return _native_outline(
        inpt,
        _torchfont.randomize_subpath_start_points,
        selection_values.cpu().contiguous().numpy(),
    )


def reorder_subpaths(inpt: Outline, keys: Tensor) -> Outline:
    """Order subpaths by explicit sort keys."""
    return _native_outline(
        inpt,
        _torchfont.randomize_subpath_order,
        keys.cpu().contiguous().numpy(),
    )


__all__ = [
    "normalize_subpath_start_points",
    "reorder_subpaths",
    "set_subpath_start_points",
]
