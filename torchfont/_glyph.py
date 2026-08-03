"""Semantic glyph references, samples, and transformed payloads."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from torch.utils._pytree import register_pytree_node

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torchfont._font import FontRef, VariationLocation

T = TypeVar("T")


@dataclass(frozen=True)
class GlyphRef:
    """Reference to one glyph face and codepoint before choosing a location."""

    font: FontRef
    codepoint: int


@dataclass(frozen=True)
class GlyphSample:
    """Dataset-local sample for one font face and codepoint."""

    ref: GlyphRef
    font_idx: int
    character_idx: int


@dataclass(frozen=True, eq=False)
class GlyphData(Generic[T]):
    """A loaded glyph payload together with its reference and targets."""

    data: T
    ref: GlyphRef
    location: VariationLocation
    font_idx: int
    character_idx: int
    weight: float
    width: float
    italic: float
    slant: float
    optical_size: float


def _flatten_glyph_data(value: GlyphData[Any]) -> tuple[list[Any], object]:
    return [value.data], (
        value.ref,
        value.location,
        value.font_idx,
        value.character_idx,
        value.weight,
        value.width,
        value.italic,
        value.slant,
        value.optical_size,
    )


def _unflatten_glyph_data(
    children: Iterable[Any], context: tuple[Any, ...]
) -> GlyphData[Any]:
    (data,) = children
    return GlyphData(data, *context)


register_pytree_node(GlyphData, _flatten_glyph_data, _unflatten_glyph_data)


__all__ = ["GlyphData", "GlyphRef", "GlyphSample"]
