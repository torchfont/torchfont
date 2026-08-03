"""Functional glyph loading operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchfont import _torchfont
from torchfont.structures import (
    COORD_DIM,
    GlyphRef,
    Outline,
    VariationLocation,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


def load_glyph(
    ref: GlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline:
    """Load one glyph outline at an explicit or default location."""
    normalized_location = (
        None if location is None else dict(VariationLocation(location))
    )
    raw_types, raw_coords = _torchfont.load_glyph(
        ref.font.path,
        ref.font.ttc_index,
        ref.codepoint,
        normalized_location,
    )
    return Outline(
        torch.from_numpy(raw_types),
        torch.from_numpy(raw_coords).view(-1, COORD_DIM),
    )


__all__ = ["load_glyph"]
