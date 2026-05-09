"""Utility functions for polishing glyph tensors before training."""

from __future__ import annotations

import torch
from torch import Tensor

from torchfont.io import CommandType


def QuadToCubic(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:  # noqa: N802
    """Convert ``CommandType.QUAD_TO`` entries to ``CommandType.CURVE_TO``.

    The command and coordinate shapes are preserved. Coordinate rows use the
    ``[cx0, cy0, cx1, cy1, x, y]`` layout, with quadratic control points read
    from ``[cx0, cy0]`` and endpoints from ``[x, y]``.
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

    q_prev = prev[quad]
    q_ctrl = out_coords[quad, 0:2]
    q_end = out_coords[quad, 4:6]

    out_coords[quad, 0:2] = q_prev + (2.0 / 3.0) * (q_ctrl - q_prev)
    out_coords[quad, 2:4] = q_end + (2.0 / 3.0) * (q_ctrl - q_end)
    out_types[quad] = CommandType.CURVE_TO.value

    return out_types, out_coords


__all__ = ["QuadToCubic"]
