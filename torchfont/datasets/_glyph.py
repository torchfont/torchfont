"""Font-face glyph dataset."""

from __future__ import annotations

from bisect import bisect_right
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, cast, overload

from torchfont._glyph import GlyphRef, GlyphSample
from torchfont.datasets._base import _BaseGlyphDataset
from torchfont.datasets._utils import normalize_index

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from pathlib import Path

T = TypeVar("T")


class GlyphDataset(_BaseGlyphDataset[T], Generic[T]):
    """Map-style dataset yielding one sample per font face and codepoint."""

    @overload
    def __init__(
        self: GlyphDataset[GlyphSample],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: str | Sequence[str] | None = None,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: GlyphDataset[T],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: str | Sequence[str] | None = None,
        transform: Callable[[GlyphSample], T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: str | Sequence[str] | None = None,
        transform: Callable[[GlyphSample], T] | None = None,
    ) -> None:
        super().__init__(
            root,
            codepoints=codepoints,
            patterns=patterns,
            transform=cast("Callable[[object], T] | None", transform),
        )

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(root={str(self.root)!r}, samples={len(self)}, "
            f"font_classes={len(self.font_classes)}, "
            f"character_classes={len(self.character_classes)})"
        )

    @overload
    def __getitem__(
        self: GlyphDataset[GlyphSample], idx: SupportsIndex
    ) -> GlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...

    def __getitem__(self, idx: SupportsIndex) -> T:
        sample_idx = normalize_index(idx, len(self))
        font_idx = bisect_right(self._offsets, sample_idx) - 1
        character_idx = int(self._character_index[sample_idx])
        sample = GlyphSample(
            ref=GlyphRef(
                self._font_refs[font_idx],
                int(self._character_codepoints[character_idx]),
            ),
            font_idx=font_idx,
            character_idx=character_idx,
        )
        return (
            self.transform(sample) if self.transform is not None else cast("T", sample)
        )


__all__ = ["GlyphDataset"]
