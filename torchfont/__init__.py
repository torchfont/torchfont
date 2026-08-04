"""TorchFont: A PyTorch-native toolkit for modeling and processing vector fonts.

Notes:
    TorchFont is local-first. You point it at a directory of font files or an
    already-synced repository checkout on disk, and it returns glyph samples
    suitable for PyTorch training code.

Features:
    * A primary ``GlyphDataset`` API for local font directories and checkouts.
    * A Rust backend that renders glyph outlines directly into PyTorch-ready
      tensors.
    * Semantic, composable transform pipelines for adapting glyph samples.

Examples:
    Assemble a dataset from local fonts::

        from torchfont.datasets import GlyphDataset

        ds = GlyphDataset(root="~/fonts")

References:
    The project README covers installation, advanced usage, and contribution
    guidelines in greater depth.

Package Layout:
    Core data types are available directly from ``torchfont``. Other public
    APIs live in submodules such as ``torchfont.datasets``,
    ``torchfont.glyphsets``, ``torchfont.nn`` and ``torchfont.transforms``.

"""

from torchfont._font import FontRef, VariationLocation
from torchfont._glyph import GlyphData, GlyphRef, GlyphSample
from torchfont._outline import COORD_DIM, TYPE_DIM, ElementType, Outline

from . import datasets, glyphsets, nn, transforms

__all__ = [
    "COORD_DIM",
    "TYPE_DIM",
    "ElementType",
    "FontRef",
    "GlyphData",
    "GlyphRef",
    "GlyphSample",
    "Outline",
    "VariationLocation",
    "datasets",
    "glyphsets",
    "nn",
    "transforms",
]
