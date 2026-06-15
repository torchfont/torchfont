from collections.abc import Sequence
from typing import Literal, TypeAlias

import numpy as np

_BitmapMode: TypeAlias = Literal["fixed", "bbox", "bbox_square"]
_FillRule: TypeAlias = Literal["winding", "even_odd"]

def cubic_to_quad(
    types: np.ndarray, coords: np.ndarray
) -> tuple[list[int], list[float]]: ...
def merge_curves(
    types: np.ndarray, coords: np.ndarray
) -> tuple[list[int], list[float]]: ...
def render_bitmap(
    types: np.ndarray,
    coords: np.ndarray,
    size: int,
    mode: _BitmapMode,
    fill_rule: _FillRule,
) -> tuple[np.ndarray, int, int]: ...
def normalize_subpath_start_points(
    types: np.ndarray, coords: np.ndarray
) -> tuple[list[int], list[float]]: ...
def randomize_subpath_start_points(
    types: np.ndarray, coords: np.ndarray, random_values: np.ndarray
) -> tuple[list[int], list[float]]: ...
def reverse_closed_subpaths(
    types: np.ndarray, coords: np.ndarray
) -> tuple[list[int], list[float]]: ...
def remove_overlaps(
    types: np.ndarray, coords: np.ndarray
) -> tuple[list[int], list[float]]: ...
def quad_to_cubic(
    types: np.ndarray, coords: np.ndarray, merge_curves: bool
) -> tuple[list[int], list[float]]: ...
def tight_bbox(
    types: np.ndarray, coords: np.ndarray
) -> tuple[float, float, float, float] | None: ...

class GlyphItem:
    types: np.ndarray
    coords: np.ndarray
    style_idx: int
    content_idx: int
    head: np.ndarray
    hhea: np.ndarray
    os2: np.ndarray
    post: np.ndarray
    maxp: np.ndarray
    hmtx: np.ndarray
    bounds: np.ndarray
    name: dict[str, str]
    codepoint: int
    glyph_name: str

class GlyphDatasetBackend:
    def __init__(
        self,
        root: str,
        codepoints: Sequence[int] | None = ...,
        patterns: Sequence[str] | None = ...,
    ) -> None: ...

    sample_count: int
    style_class_count: int
    content_class_count: int
    fingerprint: int

    def content_metadata_rows(self) -> list[tuple[str, str, int]]: ...
    def style_metadata_rows(self) -> list[tuple[str, str]]: ...
    style_axes: list[list[tuple[str, float]]]
    def item(self, idx: int) -> GlyphItem: ...
    def targets(self) -> np.ndarray: ...

LATIN_CORE: list[int]
LATIN_KERNEL: list[int]

def get_glyphset_codepoints(glyphset_name: str) -> list[int]: ...
