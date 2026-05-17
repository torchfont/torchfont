"""Sequence patching for transformer-style model inputs."""

from __future__ import annotations

import torch
from torch import Tensor


def patchify(types: Tensor, coords: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Pad and reshape a glyph sequence into uniform, fixed-size patches.

    Pads ``types`` and ``coords`` with zeros to the nearest multiple of
    ``patch_size``, then splits into contiguous patches along the leading
    sequence dimension.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.
        patch_size: Number of time steps per patch. Must be >= 1.

    Returns:
        Tuple ``(patch_types, patch_coords)`` where ``patch_types`` has shape
        ``(num_patches, patch_size)`` and ``patch_coords`` has shape
        ``(num_patches, patch_size, 6)``.

    """
    if patch_size < 1:
        msg = "patch_size must be >= 1"
        raise ValueError(msg)
    seq_len = types.size(0)
    pad = (-seq_len) % patch_size
    num_patches = (seq_len + pad) // patch_size
    pad_types = torch.cat([types, types.new_zeros(pad)], 0)
    pad_coords = torch.cat([coords, coords.new_zeros(pad, coords.size(1))], 0)
    return pad_types.view(num_patches, patch_size), pad_coords.view(
        num_patches, patch_size, coords.size(1)
    )


__all__ = [
    "patchify",
]
