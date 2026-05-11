"""Utility functions for polishing glyph tensors before training."""

from __future__ import annotations

from typing import Literal

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont.io import CommandType

BitmapMode = Literal["fixed", "bbox", "bbox_square"]


def quad_to_cubic(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Convert ``CommandType.QUAD_TO`` entries to ``CommandType.CURVE_TO``.

    The command and coordinate shapes are preserved. Coordinate rows use the
    ``[cx0, cy0, cx1, cy1, x, y]`` layout, with quadratic control points read
    from ``[cx0, cy0]`` and endpoints from ``[x, y]``.

    The last dimension of ``types`` is treated as the sequence dimension. Any
    leading dimensions are independent sequences, so call this before chunking a
    continuous outline if endpoint continuity must cross chunk boundaries.
    """
    if not torch.any(types == CommandType.QUAD_TO.value):
        return types, coords

    seq_len = types.size(-1)
    out_types = types.cpu().contiguous().clone()
    out_coords = coords.cpu().contiguous().clone()
    _torchfont.quad_to_cubic_inplace(
        out_types.reshape(-1).numpy(),
        out_coords.reshape(-1).numpy(),
        seq_len,
    )
    return out_types, out_coords


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


def render_bitmap(
    types: Tensor, coords: Tensor, size: int = 64, mode: BitmapMode = "bbox_square"
) -> Tensor:
    """Render a glyph outline to a greyscale bitmap tensor.

    The glyph is placed with a fixed 4-pixel padding on each side. ``mode``
    controls how outline coordinates are mapped to the output bitmap.

    Args:
        types: 1-D ``torch.int64`` tensor of pen command types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)`` holding
            UPM-normalised coordinate data for each command.
        size: Output image side length in pixels for ``"fixed"`` and
            ``"bbox_square"``. For ``"bbox"``, this sets the coordinate scale
            using the same fixed ``[-0.25, 1.25]`` range, then crops the output to
            the tight glyph bounding box. Must be between 1 and 4096.
        mode: Coordinate mapping mode. ``"fixed"`` maps the fixed UPM-normalised
            range ``[-0.25, 1.25] x [-0.25, 1.25]`` to the canvas. ``"bbox"`` scales
            with the fixed-mode scale and returns a variable-size bitmap
            cropped to the tight glyph bounding box. ``"bbox_square"`` scales
            the tight glyph bounding box uniformly and centres it.

    Returns:
        uint8 tensor with values in ``[0, 255]``. Shape is ``(size, size)`` for
        ``"fixed"`` and ``"bbox_square"``, and variable ``(height, width)`` for
        ``"bbox"``.

    """
    types_c = types.cpu().contiguous()
    coords_c = coords.cpu().contiguous()
    raw, width, height = _torchfont.render_bitmap(
        types_c.numpy(), coords_c.reshape(-1).numpy(), size, mode
    )
    return torch.frombuffer(bytearray(raw), dtype=torch.uint8).view(height, width)


__all__ = ["BitmapMode", "patchify", "quad_to_cubic", "render_bitmap"]
