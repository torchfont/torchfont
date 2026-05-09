"""Utility functions for transforming glyph tensors before training.

Notes:
    Transform functions are intentionally stateless.  Combine them inside your
    own dataset transform, collate function, or training step depending on how
    much data you want to materialize.

Examples:
    Define a small sample transform yourself::

        import dataclasses
        from torchfont.transforms import patchify

        def transform(sample):
            types, coords = sample.types[:256], sample.coords[:256]
            types, coords = patchify(types, coords, 32)
            return dataclasses.replace(sample, types=types, coords=coords)

"""

from __future__ import annotations

import torch
from torch import Tensor

from torchfont.io import CommandType


def quad_to_cubic(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Convert quadratic segments into cubic segments in tensor form."""
    flat_types = types.reshape(-1)
    flat_coords = coords.reshape(-1, coords.size(-1))
    quad = flat_types == CommandType.QUAD_TO.value

    if not torch.any(quad):
        return types, coords

    out_types = flat_types.clone()
    out_coords = flat_coords.clone()

    # In valid outline streams, the previous command endpoint is the current
    # point for a quadratic segment.
    prev = torch.zeros_like(out_coords[:, 0:2])
    prev[1:] = out_coords[:-1, 4:6]

    q_prev = prev[quad]
    q_ctrl = out_coords[quad, 0:2]
    q_end = out_coords[quad, 4:6]

    out_coords[quad, 0:2] = q_prev + (2.0 / 3.0) * (q_ctrl - q_prev)
    out_coords[quad, 2:4] = q_end + (2.0 / 3.0) * (q_ctrl - q_end)
    out_types[quad] = CommandType.CURVE_TO.value

    return out_types.view_as(types), out_coords.view_as(coords)


def limit_sequence_length(
    types: Tensor, coords: Tensor, max_len: int
) -> tuple[Tensor, Tensor]:
    """Truncate glyph sequences to a maximum length."""
    if max_len < 0:
        msg = "max_len must be >= 0"
        raise ValueError(msg)
    return types[:max_len], coords[:max_len]


def patchify(types: Tensor, coords: Tensor, patch_size: int) -> tuple[Tensor, Tensor]:
    """Pad a glyph sequence and reshape it into fixed-size patches."""
    if patch_size < 1:
        msg = "patch_size must be >= 1"
        raise ValueError(msg)

    seq_len = types.size(0)
    pad = (-seq_len) % patch_size
    num_patches = (seq_len + pad) // patch_size

    pad_types = torch.cat([types, types.new_zeros(pad)], 0)
    pad_coords = torch.cat([coords, coords.new_zeros(pad, coords.size(1))], 0)

    patch_types = pad_types.view(num_patches, patch_size)
    patch_coords = pad_coords.view(num_patches, patch_size, coords.size(1))

    return patch_types, patch_coords


__all__ = [
    "limit_sequence_length",
    "patchify",
    "quad_to_cubic",
]
