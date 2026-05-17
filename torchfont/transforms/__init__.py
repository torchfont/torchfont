"""Glyph outline transform utilities.

All functions accept ``(types, coords)`` and return a transformed
``(types, coords)`` pair without modifying the inputs.
"""

from torchfont.transforms.curves import (
    cubic_to_quad,
    merge_curves,
    quad_to_cubic,
    remove_overlaps,
)
from torchfont.transforms.patch import patchify
from torchfont.transforms.bitmap import BitmapMode, render_bitmap

__all__ = [
    "BitmapMode",
    "cubic_to_quad",
    "merge_curves",
    "patchify",
    "quad_to_cubic",
    "remove_overlaps",
    "render_bitmap",
]
