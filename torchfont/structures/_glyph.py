"""Semantic glyph references, samples, and transformed payloads."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Generic, TypeVar, cast

from torch.utils._pytree import register_pytree_node

if TYPE_CHECKING:
    from os import PathLike

T = TypeVar("T")


@dataclass(frozen=True, eq=False)
class VariationLocation(Mapping[str, float]):
    """An immutable, deterministic mapping of OpenType axis tags to values."""

    _items: tuple[tuple[str, float], ...]

    def __init__(
        self,
        values: Mapping[str, float] | Iterable[tuple[str, float]] = (),
    ) -> None:
        items = (
            cast("Mapping[str, float]", values).items()
            if isinstance(values, Mapping)
            else values
        )
        normalized_values: dict[str, float] = {}
        for raw_tag, raw_value in items:
            tag = str(raw_tag)
            if tag in normalized_values:
                msg = f"duplicate variation axis tag {tag!r}"
                raise ValueError(msg)
            normalized_values[tag] = float(raw_value)
        normalized = tuple(sorted(normalized_values.items()))
        object.__setattr__(self, "_items", normalized)

    def __getitem__(self, tag: str) -> float:
        for candidate, value in self._items:
            if candidate == tag:
                return value
        raise KeyError(tag)

    def __iter__(self) -> Iterator[str]:
        return (tag for tag, _value in self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Mapping):
            return NotImplemented
        return dict(self.items()) == dict(other.items())

    def __hash__(self) -> int:
        return hash(self._items)


@dataclass(frozen=True)
class FontRef:
    """Persistent file-local reference to one font."""

    path: str
    ttc_index: int

    def __init__(self, path: str | PathLike[str], ttc_index: int) -> None:
        object.__setattr__(self, "path", os.fspath(Path(path)))
        object.__setattr__(self, "ttc_index", ttc_index)


@dataclass(frozen=True)
class GlyphRef:
    """Reference to one concrete glyph instance."""

    font: FontRef
    codepoint: int
    location: VariationLocation

    def __init__(
        self,
        font: FontRef,
        codepoint: int,
        location: (
            Mapping[str, float] | Iterable[tuple[str, float]] | VariationLocation
        ),
    ) -> None:
        object.__setattr__(self, "font", font)
        object.__setattr__(self, "codepoint", codepoint)
        object.__setattr__(
            self,
            "location",
            location
            if isinstance(location, VariationLocation)
            else VariationLocation(location),
        )


@dataclass(frozen=True)
class VariableGlyphRef:
    """Reference to one glyph before choosing a variation location."""

    font: FontRef
    codepoint: int


@dataclass(frozen=True)
class GlyphSample:
    """Dataset-local sample for a concrete glyph instance."""

    ref: GlyphRef
    font_idx: int
    style_idx: int
    character_idx: int
    weight: float | None
    width: float | None
    italic: float | None
    slant: float | None
    optical_size: float | None


@dataclass(frozen=True)
class VariableGlyphSample:
    """Dataset-local sample for a glyph whose location is chosen by transform."""

    ref: VariableGlyphRef
    font_idx: int
    character_idx: int


@dataclass(frozen=True, eq=False)
class GlyphData(Generic[T]):
    """A transformed glyph payload together with its dataset metadata."""

    data: T
    sample: GlyphSample | VariableGlyphSample
    location: VariationLocation

    def __init__(
        self,
        data: T,
        sample: GlyphSample | VariableGlyphSample,
        location: (
            Mapping[str, float] | Iterable[tuple[str, float]] | VariationLocation
        ),
    ) -> None:
        object.__setattr__(self, "data", data)
        object.__setattr__(self, "sample", sample)
        object.__setattr__(
            self,
            "location",
            location
            if isinstance(location, VariationLocation)
            else VariationLocation(location),
        )


def _flatten_glyph_data(value: GlyphData[Any]) -> tuple[list[Any], object]:
    return [value.data], (value.sample, value.location)


def _unflatten_glyph_data(
    children: Iterable[Any], context: tuple[Any, Any]
) -> GlyphData[Any]:
    sample, location = context
    (data,) = children
    return GlyphData(data, sample, location)


register_pytree_node(GlyphData, _flatten_glyph_data, _unflatten_glyph_data)


__all__ = [
    "FontRef",
    "GlyphData",
    "GlyphRef",
    "GlyphSample",
    "VariableGlyphRef",
    "VariableGlyphSample",
    "VariationLocation",
]
