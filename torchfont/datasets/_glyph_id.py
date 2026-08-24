"""Glyph-id-indexed font dataset."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, cast, overload

import numpy as np
import torch
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont._font import FontRef
from torchfont._glyph import GlyphIdSample, GlyphRef
from torchfont.datasets._utils import (
    font_targets_from_offsets,
    normalize_index,
    normalize_patterns,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt
    from torch import Tensor

T = TypeVar("T")


class GlyphIdDataset(Dataset[T], Generic[T]):
    """Map-style dataset yielding one sample per font face and glyph id.

    Every face contributes one sample per glyph it draws an outline for,
    including ligatures, alternates, and other glyphs no codepoint maps to.
    Glyph ids are face-local, so samples carry no character target.

    Samples are laid out face by face: face ``i`` owns the half-open sample
    range ``_offsets[i]:_offsets[i + 1]``, so ``_offsets`` holds one more
    element than ``_font_refs`` and ends with the sample count. Sample ``s``
    draws glyph ``_glyph_ids[s]``.
    """

    _font_refs: tuple[FontRef, ...]
    _offsets: tuple[int, ...]
    _glyph_ids: npt.NDArray[np.uint32]

    @overload
    def __init__(
        self: GlyphIdDataset[GlyphIdSample],
        root: Path | str,
        *,
        patterns: str | Sequence[str] | None = None,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: GlyphIdDataset[T],
        root: Path | str,
        *,
        patterns: str | Sequence[str] | None = None,
        transform: Callable[[GlyphIdSample], T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        patterns: str | Sequence[str] | None = None,
        transform: Callable[[GlyphIdSample], T] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.patterns = normalize_patterns(patterns)
        (font_refs, offsets, self._glyph_ids) = _torchfont.index_glyphs(
            str(self.root), self.patterns
        )
        self._glyph_ids.flags.writeable = False
        self._font_refs = tuple(
            FontRef(path, ttc_index) for path, ttc_index in font_refs
        )
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        return self._offsets[-1]

    def __repr__(self) -> str:
        return (
            f"{type(self).__name__}(root={str(self.root)!r}, samples={len(self)}, "
            f"font_classes={len(self.font_classes)})"
        )

    @property
    def font_classes(self) -> list[FontRef]:
        """Font references sorted by dataset-local font index."""
        return list(self._font_refs)

    @property
    def font_targets(self) -> Tensor:
        """LongTensor of font target indices for each sample."""
        return font_targets_from_offsets(self._offsets)

    @property
    def glyph_ids(self) -> Tensor:
        """LongTensor of face-local glyph ids for each sample."""
        return torch.from_numpy(self._glyph_ids.astype(np.int64))

    @overload
    def __getitem__(
        self: GlyphIdDataset[GlyphIdSample], idx: SupportsIndex
    ) -> GlyphIdSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...

    def __getitem__(self, idx: SupportsIndex) -> T:
        sample_idx = normalize_index(idx, len(self))
        font_idx = bisect_right(self._offsets, sample_idx) - 1
        glyph_id: int = self._glyph_ids.item(sample_idx)
        sample = GlyphIdSample(
            ref=GlyphRef(self._font_refs[font_idx], glyph_id),
            font_idx=font_idx,
        )
        return (
            self.transform(sample) if self.transform is not None else cast("T", sample)
        )


__all__ = ["GlyphIdDataset"]
