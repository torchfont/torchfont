"""Semantic font structures shared by datasets and transforms."""

from torchfont.structures._glyph import (
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
    VariationLocation,
)
from torchfont.structures._outline import COORD_DIM, TYPE_DIM, ElementType, Outline

__all__ = [
    "COORD_DIM",
    "TYPE_DIM",
    "ElementType",
    "FontRef",
    "GlyphData",
    "GlyphRef",
    "GlyphSample",
    "Outline",
    "VariableGlyphRef",
    "VariableGlyphSample",
    "VariationLocation",
]
