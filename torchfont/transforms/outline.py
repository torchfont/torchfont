"""Whole-outline transformation functions."""

import torch
from torch import Tensor

from torchfont import _torchfont


def remove_overlaps(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Merge overlapping subpaths using Skia PathOps winding simplification.

    Args:
        types: 1-D ``torch.int64`` tensor of path element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new variable-length outline tuple ``(types, coords)`` with overlapping
        subpath edges removed when Skia PathOps can resolve the outline. If
        PathOps cannot simplify an otherwise valid outline, the original outline
        is returned unchanged.

    """
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.remove_overlaps(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.tensor(out_types, dtype=torch.long),
        torch.tensor(out_coords, dtype=torch.float32).view(-1, 6),
    )
