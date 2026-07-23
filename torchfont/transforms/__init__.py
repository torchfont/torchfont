"""Glyph sample and outline transform utilities.

Most functions accept ``(types, coords)`` and return a transformed
``(types, coords)`` pair without modifying the inputs. ``load_glyph`` is the
one bridge function: it takes a dataset glyph reference and returns the
``(types, coords)`` pair the rest of this module operates on, so it is
typically the first call inside a ``GlyphDataset``/``VariableGlyphDataset``
``transform``.
"""

from torchfont.transforms.bitmap import BitmapMode, render_bitmap
from torchfont.transforms.curves import (
    cubic_to_quad,
    merge_curves,
    quad_to_cubic,
)
from torchfont.transforms.geometric import (
    affine,
    horizontal_flip,
    random_affine,
    random_coord_jitter,
    random_horizontal_flip,
    random_vertical_flip,
    vertical_flip,
)
from torchfont.transforms.load import load_glyph, random_location
from torchfont.transforms.outline import patchify, remove_overlaps
from torchfont.transforms.subpath import (
    normalize_subpath_start_points,
    randomize_subpath_order,
    randomize_subpath_start_points,
)

__all__ = [
    "BitmapMode",
    "affine",
    "cubic_to_quad",
    "horizontal_flip",
    "load_glyph",
    "merge_curves",
    "normalize_subpath_start_points",
    "patchify",
    "quad_to_cubic",
    "random_affine",
    "random_coord_jitter",
    "random_horizontal_flip",
    "random_location",
    "random_vertical_flip",
    "randomize_subpath_order",
    "randomize_subpath_start_points",
    "remove_overlaps",
    "render_bitmap",
    "vertical_flip",
]
