"""Shared map-style glyph dataset behavior."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchfont._font import FontRef
from torchfont.datasets._utils import normalize_codepoints, normalize_patterns

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import SupportsIndex

    from torchfont import _torchfont

T = TypeVar("T")


class _BaseGlyphDataset(Dataset[T], Generic[T]):
    """Common configuration and targets for map-style glyph datasets."""

    _index: _torchfont.GlyphIndex

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None,
        patterns: str | Sequence[str] | None,
        transform: Callable[[object], T] | None,
    ) -> None:
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.patterns = normalize_patterns(patterns)
        self.codepoints = normalize_codepoints(codepoints)

    def __len__(self) -> int:
        return int(self._index.sample_count)

    @property
    def font_classes(self) -> list[FontRef]:
        """Font references sorted by dataset-local font index."""
        return [
            FontRef(path=os.fspath(path), ttc_index=ttc_index)
            for path, ttc_index in self._index.font_refs()
        ]

    @property
    def character_classes(self) -> list[str]:
        """Unicode characters sorted by dataset-local character index."""
        return [chr(codepoint) for codepoint in self._index.character_codepoints()]

    @property
    def character_class_to_idx(self) -> dict[str, int]:
        """Map Unicode characters to dataset-local class indices."""
        return {char: idx for idx, char in enumerate(self.character_classes)}

    @property
    def font_targets(self) -> Tensor:
        """LongTensor of font target indices for each sample."""
        return torch.from_numpy(self._index.font_targets())

    @property
    def character_targets(self) -> Tensor:
        """LongTensor of character target indices for each sample."""
        return torch.from_numpy(self._index.character_targets())
