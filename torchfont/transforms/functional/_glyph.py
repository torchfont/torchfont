from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import _torchfont
from torchfont._ops import _restore
from torchfont._outline import Outline

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torchfont._glyph import GlyphRef


def load_glyph(
    ref: GlyphRef,
    location: Mapping[str, float] | None = None,
) -> Outline:
    """Load one glyph outline at a default or clamped explicit location.

    Unknown axes are ignored and provided values must be finite.
    """
    normalized_location = (
        None
        if location is None
        else {str(tag): float(value) for tag, value in location.items()}
    )
    (raw_types, raw_coords), _, _ = _torchfont.load_glyph(
        ref.font.path,
        ref.font.face_index,
        ref.glyph_id,
        normalized_location,
    )
    return Outline(*_restore(raw_types, raw_coords))


__all__ = ["load_glyph"]
