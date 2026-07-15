"""Whole-outline transformation functions."""

import torch
from torch import Tensor

from torchfont import _torchfont


def patchify(types: Tensor, coords: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Pad and reshape a glyph sequence into uniform, fixed-size patches.

    Pads ``types`` and ``coords`` with zeros to the nearest multiple of
    ``patch_size``, then splits into contiguous patches along the leading
    sequence dimension.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        patch_size: Number of time steps per patch.

    Returns:
        Tuple ``(patch_types, patch_coords)`` where ``patch_types`` has shape
        ``(num_patches, patch_size)`` and ``patch_coords`` has shape
        ``(num_patches, patch_size, 6)``.

    """
    seq_len = types.size(0)
    pad = (-seq_len) % patch_size
    num_patches = (seq_len + pad) // patch_size
    pad_types = torch.cat([types, types.new_zeros(pad)], 0)
    pad_coords = torch.cat([coords, coords.new_zeros(pad, coords.size(1))], 0)
    return pad_types.view(num_patches, patch_size), pad_coords.view(
        num_patches, patch_size, coords.size(1)
    )


def remove_overlaps(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
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
