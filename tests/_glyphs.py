"""Glyph id lookup for tests that build a :class:`torchfont.GlyphRef` by hand.

``GlyphRef`` names a glyph by id. ``CodepointDataset`` resolves ids from codepoints
while indexing, so tests that skip the dataset resolve them here with fontTools
instead.
"""

from __future__ import annotations

from fontTools.ttLib import TTFont


def glyph_id(path: str, char: str, face_index: int = 0) -> int:
    """Return the glyph id one face maps ``char`` to."""
    font = TTFont(path, fontNumber=face_index)
    cmap = font.getBestCmap()
    assert cmap is not None
    return font.getGlyphID(cmap[ord(char)])


def glyph_id_by_name(path: str, name: str, face_index: int = 0) -> int:
    """Return the id of one named glyph, mapped or not."""
    font = TTFont(path, fontNumber=face_index)
    return font.getGlyphID(name)
