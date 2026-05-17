"""Glyph outline rasterization."""

from typing import Literal

import torch
from torch import Tensor

from torchfont import _torchfont

BitmapMode = Literal["fixed", "bbox", "bbox_square"]


def render_bitmap(
    types: Tensor, coords: Tensor, size: int = 64, mode: BitmapMode = "bbox_square"
) -> Tensor:
    """Render a glyph outline to a greyscale bitmap tensor.

    ``mode`` controls how outline coordinates are mapped to the output bitmap.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)`` holding
            UPM-normalised coordinates for each path element.
        size: Output image side length in pixels for ``"fixed"`` and
            ``"bbox_square"``. For ``"bbox"``, this sets the `coords` scale
            using the same fixed ``[-0.25, 1.25]`` range, then crops the output to
            the tight glyph bounding box. Must be between 1 and 4096.
        mode: `coords` mapping mode. ``"fixed"`` maps the fixed UPM-normalised
            range ``[-0.25, 1.25] x [-0.25, 1.25]`` to the canvas. ``"bbox"`` scales
            with the fixed-mode scale and returns a variable-size bitmap
            cropped to the tight glyph bounding box. ``"bbox_square"`` scales
            the tight glyph bounding box uniformly and centres it.

    Returns:
        uint8 tensor with values in ``[0, 255]``. Shape is ``(size, size)`` for
        ``"fixed"`` and ``"bbox_square"``, and variable ``(height, width)`` for
        ``"bbox"``.

    """
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    raw, width, height = _torchfont.render_bitmap(
        types.numpy(), coords.reshape(-1).numpy(), size, mode
    )
    if width == 0 or height == 0:
        return torch.empty((height, width), dtype=torch.uint8)
    return torch.from_numpy(raw).view(height, width)
