"""Glyph outline transform utilities.

All functions accept ``(types, coords)`` and return a transformed
``(types, coords)`` pair without modifying the inputs.
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
from torchfont.transforms.outline import patchify, remove_overlaps

__all__ = [
    "BitmapMode",
    "affine",
    "cubic_to_quad",
    "horizontal_flip",
    "merge_curves",
    "patchify",
    "quad_to_cubic",
    "random_affine",
    "random_coord_jitter",
    "random_horizontal_flip",
    "random_vertical_flip",
    "remove_overlaps",
    "render_bitmap",
    "vertical_flip",
]
