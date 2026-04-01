"""Dataset utilities that turn font files into model-ready glyph samples.

Examples:
    Mirror the Google Fonts index into a training dataset::

        from torchfont.datasets import GoogleFonts

        ds = GoogleFonts(root="data/google/fonts", ref="main", download=True)

"""

from torchfont.datasets.folder import ContentLabel, FontFolder, StyleLabel
from torchfont.datasets.google_fonts import GoogleFonts
from torchfont.datasets.repo import FontRepo

__all__ = [
    "ContentLabel",
    "FontFolder",
    "FontRepo",
    "GoogleFonts",
    "StyleLabel",
]
