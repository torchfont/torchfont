"""Glyph id lookup for tests that build a :class:`torchfont.GlyphRef` by hand."""

from __future__ import annotations

from fontTools.ttLib import TTFont


def glyph_id(path: str, char: str, face_index: int = 0) -> int:
    font = TTFont(path, fontNumber=face_index)
    cmap = font.getBestCmap()
    assert cmap is not None
    return font.getGlyphID(cmap[ord(char)])


def glyph_id_by_name(path: str, name: str, face_index: int = 0) -> int:
    font = TTFont(path, fontNumber=face_index)
    return font.getGlyphID(name)
