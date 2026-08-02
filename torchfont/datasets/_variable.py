"""Variable-location glyph dataset."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, cast, overload

from torchfont import _torchfont
from torchfont import instance_fn as _instance_fn
from torchfont.datasets._base import _BaseGlyphDataset
from torchfont.datasets._utils import normalize_index
from torchfont.structures import FontRef, VariableGlyphRef, VariableGlyphSample

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

    from torchfont.instance_fn import InstanceCountFn

T = TypeVar("T")


class VariableGlyphDataset(_BaseGlyphDataset[T], Generic[T]):
    """Map-style dataset yielding glyph references without fixed locations."""

    _index: _torchfont.VariableGlyphIndex

    @overload
    def __init__(
        self: VariableGlyphDataset[VariableGlyphSample],
        root: Path | str,
        *,
        instance_fn: InstanceCountFn = _instance_fn.named_instance_count,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: VariableGlyphDataset[T],
        root: Path | str,
        *,
        instance_fn: InstanceCountFn = _instance_fn.named_instance_count,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: Callable[[VariableGlyphSample], T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        instance_fn: InstanceCountFn = _instance_fn.named_instance_count,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: Callable[[VariableGlyphSample], T] | None = None,
    ) -> None:
        super().__init__(
            root,
            codepoints=codepoints,
            patterns=patterns,
            transform=cast("Callable[[object], T] | None", transform),
        )
        self._index = _torchfont.VariableGlyphIndex.from_root(
            str(self.root), self.codepoints, self.patterns, instance_fn
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(root={str(self.root)!r}, samples={len(self)}, "
            f"font_classes={len(self.font_classes)}, "
            f"character_classes={len(self.character_classes)})"
        )

    @overload
    def __getitem__(
        self: VariableGlyphDataset[VariableGlyphSample], idx: SupportsIndex
    ) -> VariableGlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...

    def __getitem__(self, idx: SupportsIndex) -> T:
        return self._prepare_sample(self._index.locate(normalize_index(idx, len(self))))

    def _prepare_sample(self, located: tuple[Path, int, int, int, int]) -> T:
        path, ttc_index, font_idx, codepoint, character_idx = located
        sample = VariableGlyphSample(
            ref=VariableGlyphRef(FontRef(os.fspath(path), ttc_index), codepoint),
            font_idx=font_idx,
            character_idx=character_idx,
        )
        return (
            self.transform(sample) if self.transform is not None else cast("T", sample)
        )


__all__ = ["VariableGlyphDataset"]
