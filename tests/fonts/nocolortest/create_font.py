"""Regenerate NoOutlines-Regular.ttf.

The font maps 'A' (U+0041) via cmap but contains no outline table (no glyf,
no CFF, no CFF2).  That makes skrifa's ``outline_glyphs().get()`` return
``None`` for every glyph, so the dataset filters them all out and
``len(dataset) == 0``.

All metadata tables required by ``parse_font_tables`` (head, hhea, OS/2,
post, maxp) are present so that ``FontEntry`` construction itself succeeds.

Usage::

    uv run python tests/fonts/nocolortest/create_font.py
"""

from pathlib import Path

from fontTools.fontBuilder import FontBuilder
from fontTools.ttLib.tables._g_l_y_f import Glyph

UPEM = 1000
OUT = Path(__file__).with_name("NoOutlines-Regular.ttf")


def main() -> None:
    fb = FontBuilder(UPEM, isTTF=True)

    fb.setupGlyphOrder([".notdef", "A"])
    fb.setupCharacterMap({0x0041: "A"})

    # Empty glyphs — no contours.
    fb.setupGlyf({".notdef": Glyph(), "A": Glyph()})

    fb.setupHorizontalMetrics({".notdef": (500, 0), "A": (500, 0)})
    fb.setupHorizontalHeader(ascent=800, descent=-200)
    fb.setupOS2(
        sTypoAscender=800,
        sTypoDescender=-200,
        sTypoLineGap=0,
        usWinAscent=800,
        usWinDescent=200,
    )
    fb.setupPost()
    fb.setupHead(unitsPerEm=UPEM)
    fb.setupNameTable({"familyName": "TestNoOutlines", "styleName": "Regular"})

    font = fb.font

    # Drop the outline table so skrifa has no outline format to query.
    for tag in ("glyf", "loca"):
        if tag in font:
            del font[tag]

    font.save(OUT)


if __name__ == "__main__":
    main()
