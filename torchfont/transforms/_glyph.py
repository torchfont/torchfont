"""Glyph loading transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch import nn

from torchfont import _torchfont
from torchfont._glyph import (
    GlyphData,
    GlyphRef,
    GlyphSample,
    _glyph_data_targets,
)
from torchfont.transforms import functional as _functional

if TYPE_CHECKING:
    from torchfont._outline import Outline


class LoadGlyph(nn.Module):
    """Load one glyph at the default or a randomly sampled variation location."""

    def __init__(self, location: Literal["default", "random"] = "default") -> None:
        super().__init__()
        if location not in ("default", "random"):
            msg = "location must be 'default' or 'random'"
            raise ValueError(msg)
        self.location = location

    def forward(self, inpt: GlyphSample | GlyphRef) -> GlyphData | Outline:
        """Load the referenced glyph."""
        ref = inpt.ref if isinstance(inpt, GlyphSample) else inpt
        location = (
            _default_location(ref)
            if self.location == "default"
            else _random_location(ref)
        )
        outline = _functional.load_glyph(ref, location)
        if not isinstance(inpt, GlyphSample):
            return outline
        metrics = _torchfont.glyph_targets(ref.font.path, ref.font.ttc_index, location)
        return GlyphData(
            data=outline,
            ref=ref,
            location=location,
            **_glyph_data_targets(
                codepoint=inpt.codepoint,
                font_idx=inpt.font_idx,
                character_idx=inpt.character_idx,
                metrics=metrics,
            ),
        )

    def extra_repr(self) -> str:
        return f"location={self.location}"


def _random_location(ref: GlyphRef) -> dict[str, float]:
    location: dict[str, float] = {}
    for tag, minimum, _default, maximum in _torchfont.variation_axes(
        ref.font.path, ref.font.ttc_index
    ):
        location[str(tag)] = torch.empty(()).uniform_(minimum, maximum).item()
    return location


def _default_location(ref: GlyphRef) -> dict[str, float]:
    return {
        str(tag): float(default)
        for tag, _minimum, default, _maximum in _torchfont.variation_axes(
            ref.font.path, ref.font.ttc_index
        )
    }


__all__ = ["LoadGlyph"]
