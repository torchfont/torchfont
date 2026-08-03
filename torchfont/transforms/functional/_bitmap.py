"""Functional glyph rasterization kernels."""

from typing import Literal

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont._outline import Outline

BitmapMode = Literal["fixed", "bbox", "bbox_square"]
FillRule = Literal["winding", "even_odd"]


def render_bitmap(
    inpt: Outline,
    size: int = 64,
    mode: BitmapMode = "bbox_square",
    fill_rule: FillRule = "winding",
    *,
    antialias: bool = True,
) -> Tensor:
    """Render a glyph outline to a greyscale bitmap tensor.

    ``mode`` controls how outline coordinates are mapped to the output bitmap.

    Args:
        inpt: Glyph outline to render.
        size: Output image side length in pixels for ``"fixed"`` and
            ``"bbox_square"``. For ``"bbox"``, this sets the `coords` scale
            using the same fixed ``[-0.25, 1.25]`` range, then crops the output to
            the tight glyph bounding box. Must be between 1 and 4096.
        mode: `coords` mapping mode. ``"fixed"`` maps the fixed em-unit
            range ``[-0.25, 1.25] x [-0.25, 1.25]`` to the canvas. ``"bbox"`` scales
            with the fixed-mode scale and returns a variable-size bitmap
            cropped to the tight glyph bounding box. ``"bbox_square"`` scales
            the tight glyph bounding box uniformly and centres it.
        fill_rule: ``"winding"`` (non-zero) or ``"even_odd"``.
        antialias: Whether to compute partial pixel coverage along path edges.

    Returns:
        uint8 tensor with values in ``[0, 255]``. Shape is ``(size, size)`` for
        ``"fixed"`` and ``"bbox_square"``, and variable ``(height, width)`` for
        ``"bbox"``.

    """
    types = inpt.types.cpu().contiguous()
    coords = inpt.coords.cpu().contiguous()
    raw, width, height = _torchfont.render_bitmap(
        types.numpy(), coords.reshape(-1).numpy(), size, mode, fill_rule, antialias
    )
    return torch.from_numpy(raw).view(height, width)


__all__ = ["BitmapMode", "FillRule", "render_bitmap"]
