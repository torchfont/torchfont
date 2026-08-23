"""Shared map-style glyph dataset behavior."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Generic, TypeVar

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont._font import FontRef
from torchfont.datasets._utils import normalize_codepoints, normalize_patterns

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence
    from typing import SupportsIndex

    import numpy.typing as npt

T = TypeVar("T")


class _BaseGlyphDataset(Dataset[T], Generic[T]):
    """Common configuration, sample index, and targets for glyph datasets.

    Samples are laid out face by face: face ``i`` owns the half-open sample
    range ``_offsets[i]:_offsets[i + 1]``, so ``_offsets`` holds one more
    element than ``_font_refs`` and ends with the sample count. The codepoint
    of sample ``s`` is ``_character_codepoints[_character_index[s]]``.
    """

    _font_refs: tuple[FontRef, ...]
    _offsets: tuple[int, ...]
    _character_codepoints: npt.NDArray[np.uint32]
    _character_index: npt.NDArray[np.uint32]

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
        (
            font_refs,
            offsets,
            self._character_codepoints,
            self._character_index,
        ) = _torchfont.index_fonts(str(self.root), self.codepoints, self.patterns)
        self._character_codepoints.flags.writeable = False
        self._character_index.flags.writeable = False
        self._font_refs = tuple(
            FontRef(path, ttc_index) for path, ttc_index in font_refs
        )
        self._offsets = tuple(offsets)

    def __len__(self) -> int:
        return self._offsets[-1]

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
        offsets = np.asarray(self._offsets, dtype=np.int64)
        return torch.from_numpy(
            np.repeat(np.arange(len(self._font_refs), dtype=np.int64), np.diff(offsets))
        )

    @property
    def character_targets(self) -> Tensor:
        """LongTensor of character target indices for each sample."""
        return torch.from_numpy(self._character_index.astype(np.int64))
