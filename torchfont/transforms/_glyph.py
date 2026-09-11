"""Glyph loading transforms."""

from __future__ import annotations

from typing import Literal

import torch
from torch import nn

from torchfont import _torchfont
from torchfont._glyph import (
    CodepointData,
    CodepointSample,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
    _registered_axis_targets,
)
from torchfont._outline import _COORD_DIM, Outline
from torchfont.transforms import functional as _functional


class LoadGlyph(nn.Module):
    """Load one glyph at the default or a randomly sampled variation location."""

    def __init__(self, location: Literal["default", "random"] = "default") -> None:
        super().__init__()
        if location not in ("default", "random"):
            msg = "location must be 'default' or 'random'"
            raise ValueError(msg)
        self.location = location

    def forward(
        self, inpt: CodepointSample | GlyphIdSample | GlyphRef
    ) -> CodepointData | GlyphIdData | Outline:
        """Load the referenced glyph."""
        if isinstance(inpt, GlyphRef) and self.location == "default":
            return _functional.load_glyph(inpt)
        ref = inpt if isinstance(inpt, GlyphRef) else inpt.ref
        requested_location = (
            None if self.location == "default" else _random_location(ref)
        )
        (raw_types, raw_coords), location_items, axis_values = _torchfont.load_glyph(
            ref.font.path,
            ref.font.face_index,
            ref.glyph_id,
            requested_location,
        )
        outline = Outline._wrap(  # noqa: SLF001
            torch.from_numpy(raw_types),
            torch.from_numpy(raw_coords).view(-1, _COORD_DIM),
        )
        location = dict(location_items)
        if isinstance(inpt, GlyphRef):
            return outline
        if isinstance(inpt, GlyphIdSample):
            return GlyphIdData(
                data=outline,
                ref=ref,
                location=location,
                font_idx=inpt.font_idx,
                **_registered_axis_targets(axis_values),
            )
        return CodepointData(
            data=outline,
            ref=ref,
            location=location,
            codepoint=inpt.codepoint,
            font_idx=inpt.font_idx,
            character_idx=inpt.character_idx,
            **_registered_axis_targets(axis_values),
        )

    def extra_repr(self) -> str:
        return f"location={self.location}"


def _random_location(ref: GlyphRef) -> dict[str, float]:
    location: dict[str, float] = {}
    for tag, minimum, _default, maximum in _torchfont.variation_axes(
        ref.font.path, ref.font.face_index
    ):
        location[str(tag)] = torch.empty(()).uniform_(minimum, maximum).item()
    return location


__all__ = ["LoadGlyph"]
