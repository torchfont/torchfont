"""Bitmap rendering transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline
    from torchfont.transforms.functional._bitmap import BitmapMode, FillRule


class RenderBitmap(Transform):
    """Render outlines into ``uint8`` greyscale ``H x W`` bitmap tensors.

    Apply ``torchvision.transforms.v2.ToImage`` afterwards to obtain a
    channel-first ``tv_tensors.Image`` for TorchVision pipelines.
    """

    def __init__(
        self,
        size: int = 64,
        mode: BitmapMode = "bbox_square",
        fill_rule: FillRule = "winding",
        *,
        antialias: bool = True,
    ) -> None:
        super().__init__()
        self.size = size
        self.mode = mode
        self.fill_rule = fill_rule
        self.antialias = antialias

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Tensor:
        del params
        return _functional.render_bitmap(
            inpt,
            self.size,
            self.mode,
            self.fill_rule,
            antialias=self.antialias,
        )


__all__ = ["RenderBitmap"]
