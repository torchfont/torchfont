from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np

_BitmapMode: TypeAlias = Literal["fixed", "bbox", "bbox_square"]

def render_bitmap(
    types: np.ndarray, coords: np.ndarray, size: int, mode: _BitmapMode
) -> tuple[bytes, int, int]: ...
def quad_to_cubic(types: np.ndarray, coords: np.ndarray, seq_len: int) -> None: ...
def remove_overlaps(
    types: np.ndarray, coords: np.ndarray
) -> tuple[list[int], list[float]]: ...

class GlyphItem:
    types: np.ndarray
    coords: np.ndarray
    style_idx: int
    content_idx: int
    metrics: np.ndarray
    glyph_name: str

class GlyphDataset:
    def __init__(
        self,
        root: str,
        codepoints: Sequence[int] | None = ...,
        patterns: Sequence[str] | None = ...,
    ) -> None: ...

    sample_count: int
    style_class_count: int
    content_class_count: int

    def content_metadata_rows(self) -> list[tuple[str, str, int]]: ...
    def style_metadata_rows(self, root: str) -> list[tuple[str, str]]: ...
    style_axes: list[list[tuple[str, float]]]
    def item(self, idx: int) -> GlyphItem: ...
    def targets(self) -> np.ndarray: ...
