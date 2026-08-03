"""Fixed-location glyph dataset."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, cast, overload

import torch
from torch import Tensor

from torchfont import _torchfont
from torchfont import instance_fn as _instance_fn
from torchfont.datasets._base import _BaseGlyphDataset
from torchfont.datasets._utils import normalize_index
from torchfont.structures import FontRef, GlyphRef, GlyphSample

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from torchfont.instance_fn import InstanceLocationsFn

T = TypeVar("T")


class GlyphDataset(_BaseGlyphDataset[T], Generic[T]):
    """Map-style dataset yielding fixed-location glyph references."""

    _index: _torchfont.FixedGlyphIndex

    @overload
    def __init__(
        self: GlyphDataset[GlyphSample],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: str | Sequence[str] | None = None,
        instance_fn: InstanceLocationsFn = _instance_fn.named_instances,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: GlyphDataset[T],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: str | Sequence[str] | None = None,
        instance_fn: InstanceLocationsFn = _instance_fn.named_instances,
        transform: Callable[[GlyphSample], T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: str | Sequence[str] | None = None,
        instance_fn: InstanceLocationsFn = _instance_fn.named_instances,
        transform: Callable[[GlyphSample], T] | None = None,
    ) -> None:
        super().__init__(
            root,
            codepoints=codepoints,
            patterns=patterns,
            transform=cast("Callable[[object], T] | None", transform),
        )
        self._index = _torchfont.FixedGlyphIndex.from_root(
            str(self.root), self.codepoints, self.patterns, instance_fn
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(root={str(self.root)!r}, samples={len(self)}, "
            f"font_classes={len(self.font_classes)}, styles={self._index.style_count}, "
            f"character_classes={len(self.character_classes)})"
        )

    @overload
    def __getitem__(
        self: GlyphDataset[GlyphSample], idx: SupportsIndex
    ) -> GlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...

    def __getitem__(self, idx: SupportsIndex) -> T:
        return self._prepare_sample(self._index.locate(normalize_index(idx, len(self))))

    def _prepare_sample(
        self,
        located: tuple[
            Path,
            int,
            int,
            int,
            list[tuple[str, float]],
            int,
            int,
            float | None,
            float | None,
            float | None,
            float | None,
            float | None,
        ],
    ) -> T:
        (
            path,
            ttc_index,
            font_idx,
            codepoint,
            location,
            style_idx,
            character_idx,
            weight,
            width,
            italic,
            slant,
            optical_size,
        ) = located
        sample = GlyphSample(
            ref=GlyphRef(FontRef(os.fspath(path), ttc_index), codepoint, location),
            font_idx=font_idx,
            style_idx=style_idx,
            character_idx=character_idx,
            weight=weight,
            width=width,
            italic=italic,
            slant=slant,
            optical_size=optical_size,
        )
        return (
            self.transform(sample) if self.transform is not None else cast("T", sample)
        )

    @property
    def style_classes(self) -> list[str]:
        return self._index.style_classes()

    @property
    def style_targets(self) -> Tensor:
        return torch.from_numpy(self._index.style_targets())

    @property
    def weight_targets(self) -> Tensor:
        return torch.from_numpy(self._index.weight_targets())

    @property
    def width_targets(self) -> Tensor:
        return torch.from_numpy(self._index.width_targets())

    @property
    def italic_targets(self) -> Tensor:
        return torch.from_numpy(self._index.italic_targets())

    @property
    def slant_targets(self) -> Tensor:
        return torch.from_numpy(self._index.slant_targets())

    @property
    def optical_size_targets(self) -> Tensor:
        return torch.from_numpy(self._index.optical_size_targets())


__all__ = ["GlyphDataset"]
