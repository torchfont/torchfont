"""Functional glyph loading operations."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchfont import _torchfont
from torchfont._outline import COORD_DIM, Outline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torchfont._glyph import GlyphRef


def load_glyph(
    ref: GlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline:
    """Load one glyph outline at an explicit or default location."""
    normalized_location = (
        None
        if location is None
        else {str(tag): float(value) for tag, value in location.items()}
    )
    raw_types, raw_coords = _torchfont.load_glyph(
        ref.font.path,
        ref.font.ttc_index,
        ref.codepoint,
        normalized_location,
    )
    return Outline._wrap(  # noqa: SLF001
        torch.from_numpy(raw_types),
        torch.from_numpy(raw_coords).view(-1, COORD_DIM),
    )


__all__ = ["load_glyph"]
