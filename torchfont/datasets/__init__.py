"""Map-style datasets for local font collections."""

from torchfont.datasets._glyph import GlyphDataset
from torchfont.datasets._variable import VariableGlyphDataset

__all__ = ["GlyphDataset", "VariableGlyphDataset"]
