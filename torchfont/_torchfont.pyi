from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np

_BitmapMode: TypeAlias = Literal["fixed", "bbox", "bbox_square"]
_FillRule: TypeAlias = Literal["winding", "even_odd"]

def cubic_to_quad(
    types: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def merge_curves(
    types: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def random_split_segments(
    types: np.ndarray,
    coords: np.ndarray,
    selection_values: np.ndarray,
    position_values: np.ndarray,
    split_probability: float,
    split_range: tuple[float, float],
) -> tuple[np.ndarray, np.ndarray]: ...
def render_bitmap(
    types: np.ndarray,
    coords: np.ndarray,
    size: int,
    mode: _BitmapMode,
    fill_rule: _FillRule,
) -> tuple[np.ndarray, int, int]: ...
def normalize_subpath_start_points(
    types: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def randomize_subpath_order(
    types: np.ndarray, coords: np.ndarray, random_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def randomize_subpath_start_points(
    types: np.ndarray, coords: np.ndarray, random_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def reverse_closed_subpaths(
    types: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def remove_overlaps(
    types: np.ndarray, coords: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def random_remove_overlaps(
    types: np.ndarray, coords: np.ndarray, random_values: np.ndarray
) -> tuple[np.ndarray, np.ndarray]: ...
def quad_to_cubic(
    types: np.ndarray, coords: np.ndarray, merge_curves: bool
) -> tuple[np.ndarray, np.ndarray]: ...
def tight_bbox(
    types: np.ndarray, coords: np.ndarray
) -> tuple[float, float, float, float] | None: ...

class GlyphIndex:
    sample_count: int
    @classmethod
    def from_root(
        cls,
        root: str,
        codepoints: Sequence[int] | None,
        patterns: Sequence[str] | None,
    ) -> GlyphIndex: ...
    def font_refs(self) -> list[tuple[Path, int]]: ...
    def character_codepoints(self) -> list[int]: ...
    def locate(self, idx: int) -> tuple[Path, int, int, int, int]: ...
    def font_targets(self) -> np.ndarray: ...
    def character_targets(self) -> np.ndarray: ...

def load_glyph(
    path: str,
    ttc_index: int,
    codepoint: int,
    location: dict[str, float] | None = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
def variation_axes(
    path: str,
    ttc_index: int,
) -> list[tuple[str, float, float, float]]: ...

LATIN_CORE: list[int]
LATIN_KERNEL: list[int]

def get_glyphset_codepoints(glyphset_name: str) -> list[int]: ...
