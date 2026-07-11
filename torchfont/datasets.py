"""Utilities for turning local font folders into indexed glyph datasets.

Dataset indices and targets are built from font files at construction time,
while glyph outlines are loaded lazily from the current files on disk.
Modifying font files during a dataset object's lifetime, including across
pickle/unpickle boundaries, is unsupported and may produce inconsistent
samples or labels.
"""

from __future__ import annotations

import dataclasses
import os
from operator import index
from pathlib import Path
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, cast, overload

import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont import variation as _variation

_T = TypeVar("_T")
_V = TypeVar("_V")

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torchfont.variation import InstanceCountFn, InstanceFn


@dataclasses.dataclass(frozen=True)
class FontRef:
    """Persistent file-local reference to one font."""

    path: str
    ttc_index: int


@dataclasses.dataclass(frozen=True)
class GlyphRef:
    """Reference to one concrete glyph instance."""

    font: FontRef
    codepoint: int
    location: Mapping[str, float]


@dataclasses.dataclass(frozen=True)
class VariableGlyphRef:
    """Reference to one glyph before choosing a variation location."""

    font: FontRef
    codepoint: int


@dataclasses.dataclass(frozen=True)
class GlyphSample:
    """Dataset-local sample for a concrete glyph instance."""

    ref: GlyphRef
    font_idx: int
    style_idx: int
    character_idx: int


@dataclasses.dataclass(frozen=True)
class VariableGlyphSample:
    """Dataset-local sample for a glyph whose location is chosen by transform."""

    ref: VariableGlyphRef
    font_idx: int
    character_idx: int


class GlyphDataset(Dataset[_T], Generic[_T]):
    """Dataset that yields fixed-location glyph references.

    The index and targets are fixed at construction time, but glyph outlines
    are loaded lazily from the current files on disk. Do not modify indexed
    font files while a dataset object is in use.
    """

    @overload
    def __init__(
        self: GlyphDataset[GlyphSample],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        instances: InstanceFn = _variation.named_instances,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: GlyphDataset[_T],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        instances: InstanceFn = _variation.named_instances,
        transform: Callable[[GlyphSample], _T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        instances: InstanceFn = _variation.named_instances,
        transform: Callable[[GlyphSample], _T] | None = None,
    ) -> None:
        """Scan ``root`` and build fixed-location glyph sample metadata."""
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.patterns = _normalize_patterns(patterns)
        self.codepoints = _normalize_codepoints(codepoints)

        self._index = _torchfont.FixedGlyphIndex.from_root(
            str(self.root),
            self.codepoints,
            self.patterns,
            instances,
        )

    def __repr__(self) -> str:
        """Return a compact dataset summary."""
        return (
            f"{type(self).__name__}("
            f"root={str(self.root)!r}, "
            f"samples={len(self)}, "
            f"font_classes={len(self.font_classes)}, "
            f"styles={self._index.style_count}, "
            f"character_classes={len(self.character_classes)})"
        )

    def __len__(self) -> int:
        """Return the total number of indexed glyph samples."""
        return int(self._index.sample_count)

    @overload
    def __getitem__(
        self: GlyphDataset[GlyphSample],
        idx: SupportsIndex,
    ) -> GlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> _T: ...

    def __getitem__(self, idx: SupportsIndex) -> _T:
        """Return one glyph sample or the corresponding transform output."""
        (
            path,
            ttc_index,
            font_idx,
            codepoint,
            location,
            style_idx,
            character_idx,
        ) = self._index.locate(_normalize_index(idx, len(self)))
        sample = GlyphSample(
            ref=GlyphRef(
                font=FontRef(
                    path=os.fspath(path),
                    ttc_index=ttc_index,
                ),
                codepoint=codepoint,
                location=dict(location),
            ),
            font_idx=font_idx,
            style_idx=style_idx,
            character_idx=character_idx,
        )
        if self.transform is not None:
            return self.transform(sample)
        return cast("_T", sample)

    @property
    def font_classes(self) -> list[FontRef]:
        """Font references sorted by dataset-local font index."""
        return _font_classes(self._index)

    @property
    def style_classes(self) -> list[str]:
        """Style labels sorted by dataset-local style index."""
        return self._index.style_classes()

    @property
    def character_classes(self) -> list[str]:
        """Unicode characters sorted by dataset-local character index."""
        return _character_classes(self._index)

    @property
    def character_class_to_idx(self) -> dict[str, int]:
        """Mapping from Unicode character to character class index."""
        return _character_class_to_idx(self._index)

    @property
    def font_targets(self) -> Tensor:
        """LongTensor of font target indices for each sample."""
        return torch.from_numpy(self._index.font_targets())

    @property
    def style_targets(self) -> Tensor:
        """LongTensor of style target indices for each sample."""
        return torch.from_numpy(self._index.style_targets())

    @property
    def character_targets(self) -> Tensor:
        """LongTensor of character target indices for each sample."""
        return torch.from_numpy(self._index.character_targets())


class VariableGlyphDataset(Dataset[_V], Generic[_V]):
    """Dataset that yields glyph references without fixed variation locations.

    The index and targets are fixed at construction time, but glyph outlines
    are loaded lazily from the current files on disk. Do not modify indexed
    font files while a dataset object is in use.
    """

    @overload
    def __init__(
        self: VariableGlyphDataset[VariableGlyphSample],
        root: Path | str,
        *,
        instance_count: InstanceCountFn = _variation.named_instance_count,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: VariableGlyphDataset[_V],
        root: Path | str,
        *,
        instance_count: InstanceCountFn = _variation.named_instance_count,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: Callable[[VariableGlyphSample], _V],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        instance_count: InstanceCountFn = _variation.named_instance_count,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: Callable[[VariableGlyphSample], _V] | None = None,
    ) -> None:
        """Scan ``root`` and build variable-location glyph sample metadata."""
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.patterns = _normalize_patterns(patterns)
        self.codepoints = _normalize_codepoints(codepoints)

        self._index = _torchfont.VariableGlyphIndex.from_root(
            str(self.root),
            self.codepoints,
            self.patterns,
            instance_count,
        )

    def __repr__(self) -> str:
        """Return a compact dataset summary."""
        return (
            f"{type(self).__name__}("
            f"root={str(self.root)!r}, "
            f"samples={len(self)}, "
            f"font_classes={len(self.font_classes)}, "
            f"character_classes={len(self.character_classes)})"
        )

    def __len__(self) -> int:
        """Return the total number of indexed glyph samples."""
        return int(self._index.sample_count)

    @overload
    def __getitem__(
        self: VariableGlyphDataset[VariableGlyphSample],
        idx: SupportsIndex,
    ) -> VariableGlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> _V: ...

    def __getitem__(self, idx: SupportsIndex) -> _V:
        """Return one variable glyph sample or transform output."""
        path, ttc_index, font_idx, codepoint, character_idx = self._index.locate(
            _normalize_index(idx, len(self)),
        )
        sample = VariableGlyphSample(
            ref=VariableGlyphRef(
                font=FontRef(
                    path=os.fspath(path),
                    ttc_index=ttc_index,
                ),
                codepoint=codepoint,
            ),
            font_idx=font_idx,
            character_idx=character_idx,
        )
        if self.transform is not None:
            return self.transform(sample)
        return cast("_V", sample)

    @property
    def font_classes(self) -> list[FontRef]:
        """Font references sorted by dataset-local font index."""
        return _font_classes(self._index)

    @property
    def character_classes(self) -> list[str]:
        """Unicode characters sorted by dataset-local character index."""
        return _character_classes(self._index)

    @property
    def character_class_to_idx(self) -> dict[str, int]:
        """Mapping from Unicode character to character class index."""
        return _character_class_to_idx(self._index)

    @property
    def font_targets(self) -> Tensor:
        """LongTensor of font target indices for each sample."""
        return torch.from_numpy(self._index.font_targets())

    @property
    def character_targets(self) -> Tensor:
        """LongTensor of character target indices for each sample."""
        return torch.from_numpy(self._index.character_targets())


def _normalize_patterns(patterns: Sequence[str] | None) -> tuple[str, ...] | None:
    if patterns is None:
        return None
    return tuple(str(pattern) for pattern in patterns)


def _normalize_codepoints(
    codepoints: Sequence[SupportsIndex] | None,
) -> tuple[int, ...] | None:
    if codepoints is None:
        return None
    return tuple(sorted({index(codepoint) for codepoint in codepoints}))


def _normalize_index(idx: SupportsIndex, dataset_len: int) -> int:
    resolved_idx = index(idx)
    original_idx = resolved_idx
    if resolved_idx < 0:
        resolved_idx += dataset_len
    if resolved_idx < 0 or resolved_idx >= dataset_len:
        msg = (
            f"index {original_idx} is out of range for dataset of length {dataset_len}"
        )
        raise IndexError(msg)
    return resolved_idx


def _font_classes(
    index_: _torchfont.FixedGlyphIndex | _torchfont.VariableGlyphIndex,
) -> list[FontRef]:
    return [
        FontRef(path=os.fspath(path), ttc_index=ttc_index)
        for path, ttc_index in index_.font_refs()
    ]


def _character_classes(
    index_: _torchfont.FixedGlyphIndex | _torchfont.VariableGlyphIndex,
) -> list[str]:
    return [chr(codepoint) for codepoint in index_.character_codepoints()]


def _character_class_to_idx(
    index_: _torchfont.FixedGlyphIndex | _torchfont.VariableGlyphIndex,
) -> dict[str, int]:
    return {char: idx for idx, char in enumerate(_character_classes(index_))}


__all__ = [
    "FontRef",
    "GlyphDataset",
    "GlyphRef",
    "GlyphSample",
    "VariableGlyphDataset",
    "VariableGlyphRef",
    "VariableGlyphSample",
]
