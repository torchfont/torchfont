"""Functional glyph loading operations."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from torchfont import _torchfont
from torchfont.structures import COORD_DIM, GlyphRef, Outline, VariableGlyphRef

if TYPE_CHECKING:
    from collections.abc import Mapping


@overload
def load_glyph(ref: GlyphRef) -> Outline: ...


@overload
def load_glyph(
    ref: VariableGlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline: ...


def load_glyph(
    ref: GlyphRef | VariableGlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline:
    """Load one glyph outline at an explicit, deterministic location."""
    if isinstance(ref, GlyphRef):
        if location is not None:
            msg = "location cannot override a GlyphRef location"
            raise ValueError(msg)
        location = ref.location
    raw_types, raw_coords = _torchfont.load_glyph(
        ref.font.path,
        ref.font.ttc_index,
        ref.codepoint,
        None
        if location is None
        else {str(tag): float(value) for tag, value in location.items()},
    )
    return Outline(
        torch.from_numpy(raw_types),
        torch.from_numpy(raw_coords).view(-1, COORD_DIM),
    )


__all__ = ["load_glyph"]
