"""Dataset utilities that turn font files into model-ready glyph samples.

Examples:
    Load glyphs from a local font directory::

        from torchfont.datasets import GlyphDataset

        ds = GlyphDataset(root="~/fonts")

"""

from torchfont.datasets.folder import (
    ContentLabel,
    FontFolder,
    GlyphDataset,
    StyleLabel,
)
from torchfont.datasets.google_fonts import GoogleFonts
from torchfont.datasets.repo import FontRepo

__all__ = [
    "ContentLabel",
    "FontFolder",
    "FontRepo",
    "GlyphDataset",
    "GoogleFonts",
    "StyleLabel",
]
