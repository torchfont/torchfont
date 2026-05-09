"""Utility functions for polishing glyph tensors before training."""

from __future__ import annotations

import torch
from torch import Tensor

from torchfont.io import CommandType


def quad_to_cubic(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Convert ``CommandType.QUAD_TO`` entries to ``CommandType.CURVE_TO``.

    The command and coordinate shapes are preserved. Coordinate rows use the
    ``[cx0, cy0, cx1, cy1, x, y]`` layout, with quadratic control points read
    from ``[cx0, cy0]`` and endpoints from ``[x, y]``.

    The last dimension of ``types`` is treated as the sequence dimension. Any
    leading dimensions are independent sequences, so call this before chunking a
    continuous outline if endpoint continuity must cross chunk boundaries.
    """
    quad = types == CommandType.QUAD_TO.value

    if not torch.any(quad):
        return types, coords

    out_types = types.clone()
    out_coords = coords.clone()

    # In valid outline streams, the previous command endpoint is the current
    # point for a quadratic segment. Leading dimensions are independent samples.
    prev = torch.zeros_like(out_coords[..., 0:2])
    prev[..., 1:, :] = out_coords[..., :-1, 4:6]

    flat_quad = quad.reshape(-1)
    flat_types = out_types.reshape(-1)
    flat_coords = out_coords.reshape(-1, coords.size(-1))
    flat_prev = prev.reshape(-1, 2)

    q_prev = flat_prev[flat_quad]
    q_ctrl = flat_coords[flat_quad, 0:2]
    q_end = flat_coords[flat_quad, 4:6]

    flat_coords[flat_quad, 0:2] = q_prev + (2.0 / 3.0) * (q_ctrl - q_prev)
    flat_coords[flat_quad, 2:4] = q_end + (2.0 / 3.0) * (q_ctrl - q_end)
    flat_types[flat_quad] = CommandType.CURVE_TO.value

    return out_types, out_coords


__all__ = ["quad_to_cubic"]
