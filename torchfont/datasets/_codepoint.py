"""Codepoint-indexed font dataset."""

from __future__ import annotations

from bisect import bisect_right
from pathlib import Path
from typing import TYPE_CHECKING, Generic, SupportsIndex, TypeVar, cast, overload

import numpy as np
import torch
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont._font import FontRef
from torchfont._glyph import GlyphRef, GlyphSample
from torchfont.datasets._utils import (
    font_targets_from_offsets,
    normalize_codepoints,
    normalize_index,
    normalize_max_length,
    normalize_patterns,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    import numpy.typing as npt
    from torch import Tensor

T = TypeVar("T")


class CodepointDataset(Dataset[T], Generic[T]):
    """Map-style dataset yielding one sample per font face and codepoint.

    Every face contributes one sample per codepoint its ``cmap`` maps to an
    outline glyph. Glyphs no codepoint reaches, such as ligatures and
    alternates, are unreachable here; index them with
    :class:`~torchfont.datasets.GlyphIdDataset` instead.

    ``max_length`` keeps only glyphs whose outline is at most that many elements
    long.

    Samples are laid out face by face: face ``i`` owns the half-open sample
    range ``_offsets[i]:_offsets[i + 1]``, so ``_offsets`` holds one more
    element than ``_font_refs`` and ends with the sample count. Sample ``s``
    draws glyph ``_glyph_ids[s]``, whose codepoint is
    ``_character_codepoints[_character_index[s]]``.
    """

    _font_refs: tuple[FontRef, ...]
    _offsets: tuple[int, ...]
    _character_codepoints: npt.NDArray[np.uint32]
    _character_index: npt.NDArray[np.uint32]
    _glyph_ids: npt.NDArray[np.uint32]

    @overload
    def __init__(
        self: CodepointDataset[GlyphSample],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        max_length: SupportsIndex | None = None,
        patterns: str | Sequence[str] | None = None,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: CodepointDataset[T],
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        max_length: SupportsIndex | None = None,
        patterns: str | Sequence[str] | None = None,
        transform: Callable[[GlyphSample], T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        max_length: SupportsIndex | None = None,
        patterns: str | Sequence[str] | None = None,
        transform: Callable[[GlyphSample], T] | None = None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.patterns = normalize_patterns(patterns)
        self.codepoints = normalize_codepoints(codepoints)
        self.max_length = normalize_max_length(max_length)
        (
            font_refs,
            offsets,
            self._character_codepoints,
            self._character_index,
            self._glyph_ids,
        ) = _torchfont.index_codepoints(
            str(self.root), self.codepoints, self.max_length, self.patterns
        )
        self._character_codepoints.flags.writeable = False
        self._character_index.flags.writeable = False
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
            f"font_classes={len(self.font_classes)}, "
            f"character_classes={len(self.character_classes)})"
        )

    @property
    def font_classes(self) -> list[FontRef]:
        """Font references sorted by dataset-local font index."""
        return list(self._font_refs)

    @property
    def character_classes(self) -> list[str]:
        """Unicode characters sorted by dataset-local character index."""
        return [chr(codepoint) for codepoint in self._character_codepoints.tolist()]

    @property
    def character_class_to_idx(self) -> dict[str, int]:
        """Map Unicode characters to dataset-local class indices."""
        return {char: idx for idx, char in enumerate(self.character_classes)}

    @property
    def font_targets(self) -> Tensor:
        """LongTensor of font target indices for each sample."""
        return font_targets_from_offsets(self._offsets)

    @property
    def character_targets(self) -> Tensor:
        """LongTensor of character target indices for each sample."""
        return torch.from_numpy(self._character_index.astype(np.int64))

    @overload
    def __getitem__(
        self: CodepointDataset[GlyphSample], idx: SupportsIndex
    ) -> GlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> T: ...

    def __getitem__(self, idx: SupportsIndex) -> T:
        sample_idx = normalize_index(idx, len(self))
        font_idx = bisect_right(self._offsets, sample_idx) - 1
        character_idx: int = self._character_index.item(sample_idx)
        glyph_id: int = self._glyph_ids.item(sample_idx)
        codepoint: int = self._character_codepoints.item(character_idx)
        sample = GlyphSample(
            ref=GlyphRef(self._font_refs[font_idx], glyph_id),
            codepoint=codepoint,
            font_idx=font_idx,
            character_idx=character_idx,
        )
        return (
            self.transform(sample) if self.transform is not None else cast("T", sample)
        )


__all__ = ["CodepointDataset"]
