"""Functional whole-outline kernels."""

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont._outline import Outline
from torchfont.transforms.functional._utils import _native_outline


def _remove_overlaps(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Merge overlapping subpaths using Skia PathOps winding simplification.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new variable-length outline tuple ``(types, coords)`` with overlapping
        subpath edges removed when Skia PathOps can resolve the outline. If
        PathOps cannot simplify an otherwise valid outline, the original outline
        is returned unchanged.

    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.remove_overlaps(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )


def remove_overlaps(inpt: Outline) -> Outline:
    """Merge overlapping subpaths."""
    return Outline(*_remove_overlaps(inpt.types, inpt.coords))


def remove_overlap_groups(inpt: Outline, selection_values: Tensor) -> Outline:
    """Simplify overlap groups according to explicit selection values."""
    return _native_outline(
        inpt,
        _torchfont.random_remove_overlaps,
        selection_values.cpu().contiguous().numpy(),
    )


__all__ = ["remove_overlap_groups", "remove_overlaps"]
