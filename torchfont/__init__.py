"""PyTorch datasets and transforms for vector fonts.

Load local font files as glyph samples, transform their outlines, and render
PyTorch tensors with a Rust backend.

Core data types are available directly from ``torchfont``. Other public APIs
are in ``torchfont.datasets``, ``torchfont.glyphsets``, ``torchfont.nn``, and
``torchfont.transforms``.
"""

from torchfont._font import FontRef
from torchfont._glyph import (
    CodepointData,
    CodepointSample,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
)
from torchfont._outline import ElementType, Outline

from . import datasets, glyphsets, nn, transforms

__all__ = [
    "CodepointData",
    "CodepointSample",
    "ElementType",
    "FontRef",
    "GlyphIdData",
    "GlyphIdSample",
    "GlyphRef",
    "Outline",
    "datasets",
    "glyphsets",
    "nn",
    "transforms",
]
