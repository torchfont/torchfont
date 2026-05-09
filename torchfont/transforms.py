"""Utility functions for polishing glyph tensors before training."""

from __future__ import annotations

import torch
from torch import Tensor

from torchfont import _torchfont
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


def render_bitmap(types: Tensor, coords: Tensor, size: int = 64) -> Tensor:
    """Render a glyph outline to a greyscale bitmap tensor.

    The glyph is auto-scaled and centred to fill the canvas with a fixed
    4-pixel padding on each side.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)`` holding
            UPM-normalised coordinate data for each command.
        size: Output image side length in pixels (square). Must be between 1
            and 4096.

    Returns:
        uint8 tensor of shape ``(size, size)`` with values in ``[0, 255]``.

    """
    types_bytes = bytes(types.cpu().contiguous().numpy().view("uint8"))
    coords_bytes = bytes(coords.cpu().contiguous().numpy().view("uint8"))
    raw = _torchfont.render_bitmap(types_bytes, coords_bytes, size)
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(size, size)


__all__ = ["quad_to_cubic", "render_bitmap"]
