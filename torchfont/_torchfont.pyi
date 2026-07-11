from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
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

class FixedGlyphIndex:
    sample_count: int
    style_count: int
    @classmethod
    def from_root(
        cls,
        root: str,
        codepoints: Sequence[int] | None,
        patterns: Sequence[str] | None,
        instances: Callable[..., Sequence[Mapping[str, float]]],
    ) -> FixedGlyphIndex: ...
    def font_refs(self) -> list[tuple[Path, int]]: ...
    def style_classes(self) -> list[str]: ...
    def character_codepoints(self) -> list[int]: ...
    def locate(
        self,
        idx: int,
    ) -> tuple[Path, int, int, int, list[tuple[str, float]], int, int]: ...
    def font_targets(self) -> np.ndarray: ...
    def style_targets(self) -> np.ndarray: ...
    def character_targets(self) -> np.ndarray: ...

class VariableGlyphIndex:
    sample_count: int
    @classmethod
    def from_root(
        cls,
        root: str,
        codepoints: Sequence[int] | None,
        patterns: Sequence[str] | None,
        instance_count: Callable[..., int],
    ) -> VariableGlyphIndex: ...
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
def default_location_for_font(path: str, ttc_index: int) -> list[tuple[str, float]]: ...
def named_instance_locations_for_font(
    path: str,
    ttc_index: int,
) -> list[list[tuple[str, float]]]: ...
def grid_locations_for_font(
    path: str,
    ttc_index: int,
    axes: dict[str, int],
) -> list[list[tuple[str, float]]]: ...
def grid_location_count_for_font(
    path: str,
    ttc_index: int,
    axes: dict[str, int],
) -> int: ...

LATIN_CORE: list[int]
LATIN_KERNEL: list[int]

def get_glyphset_codepoints(glyphset_name: str) -> list[int]: ...
