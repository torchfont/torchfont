"""Deterministic functional kernels for semantic font data."""

from torchfont.transforms.functional._bitmap import render_bitmap
from torchfont.transforms.functional._curves import (
    cubic_to_quad,
    merge_curves,
    quad_to_cubic,
    split_segments,
)
from torchfont.transforms.functional._geometry import (
    add_coordinate_noise,
    affine,
    elastic,
    horizontal_flip,
    rotate,
    scale,
    vertical_flip,
)
from torchfont.transforms.functional._glyph import load_glyph
from torchfont.transforms.functional._outline import (
    remove_overlap_groups,
    remove_overlaps,
)
from torchfont.transforms.functional._subpath import (
    drop_subpaths,
    drop_subpaths_to_fit,
    normalize_subpath_order,
    normalize_subpath_start_points,
    reorder_subpaths,
    set_subpath_start_points,
    split_subpaths,
    truncate_subpaths,
)

__all__ = [
    "add_coordinate_noise",
    "affine",
    "cubic_to_quad",
    "drop_subpaths",
    "drop_subpaths_to_fit",
    "elastic",
    "horizontal_flip",
    "load_glyph",
    "merge_curves",
    "normalize_subpath_order",
    "normalize_subpath_start_points",
    "quad_to_cubic",
    "remove_overlap_groups",
    "remove_overlaps",
    "render_bitmap",
    "reorder_subpaths",
    "rotate",
    "scale",
    "set_subpath_start_points",
    "split_segments",
    "split_subpaths",
    "truncate_subpaths",
    "vertical_flip",
]
