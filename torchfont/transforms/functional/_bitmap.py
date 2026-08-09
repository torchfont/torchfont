"""Functional glyph rasterization kernels."""

from typing import Literal

from torch import Tensor

from torchfont import _ops
from torchfont._outline import Outline
from torchfont.transforms.functional._utils import _require_no_grad, _require_single

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

    Notes:
        Integer coverage defines no gradient, so an outline that requires grad is
        rejected.

    """
    _require_single(inpt, "render_bitmap")
    _require_no_grad(inpt, "render_bitmap")
    return _ops.render_bitmap(inpt.types, inpt.coords, size, mode, fill_rule, antialias)


__all__ = ["BitmapMode", "FillRule", "render_bitmap"]
