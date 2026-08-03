"""Persistent font references and variation locations."""

from __future__ import annotations

import os
from collections.abc import Iterable, Iterator, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from os import PathLike


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


__all__ = ["FontRef", "VariationLocation"]
