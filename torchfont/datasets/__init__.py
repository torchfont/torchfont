"""Map-style datasets for local font collections."""

from torchfont.datasets._codepoint import CodepointDataset
from torchfont.datasets._glyph_id import GlyphIdDataset

__all__ = ["CodepointDataset", "GlyphIdDataset"]
