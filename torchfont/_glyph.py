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

_METRIC_FIELDS = ("weight", "width", "italic", "slant", "optical_size")


class _MetricTargets(TypedDict):
    weight: float | None
    width: float | None
    italic: float | None
    slant: float | None
    optical_size: float | None


@dataclass(frozen=True)
class GlyphRef:
    """Reference to one glyph of one font face before choosing a location."""

    font: FontRef
    glyph_id: int


@dataclass(frozen=True)
class GlyphSample:
    """Dataset-local sample for one font face and codepoint."""

    ref: GlyphRef
    codepoint: int
    font_idx: int
    character_idx: int


@dataclass(frozen=True)
class GlyphIdSample:
    """Dataset-local sample for one font face and glyph id."""

    ref: GlyphRef
    font_idx: int


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
    codepoint: int
    font_idx: int
    character_idx: int
    weight: float | None
    width: float | None
    italic: float | None
    slant: float | None
    optical_size: float | None


@dataclass(frozen=True, eq=False)
class GlyphIdData(Generic[T]):
    """A loaded glyph payload identified by glyph id rather than codepoint.

    Carries the same reference, location, and continuous targets as
    :class:`GlyphData`, without the codepoint targets that a glyph no character
    maps to cannot have.
    """

    data: T
    ref: GlyphRef
    location: dict[str, float]
    font_idx: int
    weight: float | None
    width: float | None
    italic: float | None
    slant: float | None
    optical_size: float | None


def _register_glyph_data(
    cls: type[GlyphData[Any] | GlyphIdData[Any]],
    target_fields: tuple[str, ...],
) -> None:
    """Register one payload class as a pytree whose targets are children."""

    def flatten(value: GlyphData[Any] | GlyphIdData[Any]) -> tuple[list[Any], object]:
        children = [value.data, *(getattr(value, name) for name in target_fields)]
        return children, (value.ref, value.location)

    def unflatten(
        children: Iterable[Any], context: tuple[Any, ...]
    ) -> GlyphData[Any] | GlyphIdData[Any]:
        data, *targets = children
        ref, location = context
        return cls(data, ref, location, *targets)

    register_pytree_node(cls, flatten, unflatten)


_register_glyph_data(
    GlyphData, ("codepoint", "font_idx", "character_idx", *_METRIC_FIELDS)
)
_register_glyph_data(GlyphIdData, ("font_idx", *_METRIC_FIELDS))


def _metric_targets(
    metrics: tuple[float, float, float, float, float],
) -> _MetricTargets:
    """Build the continuous targets carried by a loaded glyph payload.

    ``metrics`` is the ``(weight, width, italic, slant, optical_size)`` tuple
    returned by the Rust ``glyph_targets`` helper, in that order.
    """
    weight, width, italic, slant, optical_size = metrics
    return {
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
    "GlyphIdData",
    "GlyphIdSample",
    "GlyphRef",
    "GlyphSample",
]
