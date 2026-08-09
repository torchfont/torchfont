"""Semantic glyph references, samples, and transformed payloads."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypedDict, TypeVar

from torch.utils._pytree import register_pytree_node

if TYPE_CHECKING:
    from collections.abc import Iterable

    from torchfont._font import FontRef

T = TypeVar("T")

_TARGET_FIELDS = (
    "font_idx",
    "character_idx",
    "weight",
    "width",
    "italic",
    "slant",
    "optical_size",
)


class _GlyphDataTargets(TypedDict):
    font_idx: int
    character_idx: int
    weight: float | None
    width: float | None
    italic: float | None
    slant: float | None
    optical_size: float | None


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
    """A loaded glyph payload together with its reference and targets.

    Indices are Python integers and continuous targets are floats. An unavailable
    continuous target is ``None``.

    ``ref`` and ``location`` are metadata that no tensor can represent. Targets
    remain pytree children so their values do not become part of its structure.
    """

    data: T
    ref: GlyphRef
    location: dict[str, float]
    font_idx: int
    character_idx: int
    weight: float | None
    width: float | None
    italic: float | None
    slant: float | None
    optical_size: float | None


def _flatten_glyph_data(value: GlyphData[Any]) -> tuple[list[Any], object]:
    children = [value.data, *(getattr(value, name) for name in _TARGET_FIELDS)]
    return children, (value.ref, value.location)


def _unflatten_glyph_data(
    children: Iterable[Any], context: tuple[Any, ...]
) -> GlyphData[Any]:
    data, *targets = children
    ref, location = context
    return GlyphData(data, ref, location, *targets)


register_pytree_node(GlyphData, _flatten_glyph_data, _unflatten_glyph_data)


def _glyph_data_targets(
    *,
    font_idx: int,
    character_idx: int,
    metrics: tuple[float, float, float, float, float],
) -> _GlyphDataTargets:
    """Build the scalar targets carried by :class:`GlyphData`.

    ``metrics`` is the ``(weight, width, italic, slant, optical_size)`` tuple
    returned by the Rust ``glyph_targets`` helper, in that order.
    """
    weight, width, italic, slant, optical_size = metrics
    return {
        "font_idx": font_idx,
        "character_idx": character_idx,
        "weight": _optional_metric(weight),
        "width": _optional_metric(width),
        "italic": _optional_metric(italic),
        "slant": _optional_metric(slant),
        "optical_size": _optional_metric(optical_size),
    }


def _optional_metric(value: float) -> float | None:
    """Represent an unavailable native metric explicitly at the Python boundary."""
    return None if math.isnan(value) else value


__all__ = [
    "GlyphData",
    "GlyphRef",
    "GlyphSample",
]
