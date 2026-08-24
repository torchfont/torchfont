from collections.abc import Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt

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
    antialias: bool,
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
def index_codepoints(
    root: str,
    codepoints: Sequence[int] | None,
    max_length: int | None,
    patterns: Sequence[str] | None,
) -> tuple[
    list[tuple[Path, int]],
    list[int],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
    npt.NDArray[np.uint32],
]: ...
def index_glyphs(
    root: str,
    max_length: int | None,
    patterns: Sequence[str] | None,
) -> tuple[
    list[tuple[Path, int]],
    list[int],
    npt.NDArray[np.uint32],
]: ...
def load_glyph(
    path: str,
    face_index: int,
    glyph_id: int,
    location: dict[str, float] | None = ...,
) -> tuple[np.ndarray, np.ndarray]: ...
def variation_axes(
    path: str,
    face_index: int,
) -> list[tuple[str, float, float, float]]: ...
def glyph_targets(
    path: str,
    face_index: int,
    location: dict[str, float],
) -> tuple[
    float,
    float,
    float,
    float,
    float,
]: ...

LATIN_CORE: list[int]
LATIN_KERNEL: list[int]

def get_glyphset_codepoints(glyphset_name: str) -> list[int]: ...
