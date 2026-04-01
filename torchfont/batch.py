"""Structured representation of a collated glyph batch."""

from typing import NamedTuple

from torch import Tensor


class GlyphBatch(NamedTuple):
    """One collated glyph batch.

    Attributes:
        types (Tensor): Long tensor of shape ``(B, L, ...)`` holding padded
            command types.
        coords (Tensor): Float tensor of shape ``(B, L, ..., 6)`` holding
            padded coordinate values.
        style_idx (Tensor): 1-D long tensor of style indices.
        content_idx (Tensor): 1-D long tensor of content indices.
        mask (Tensor): Boolean tensor marking valid, non-padding sequence
            positions. Shape is ``(B, L)``.

    """

    types: Tensor
    coords: Tensor
    style_idx: Tensor
    content_idx: Tensor
    mask: Tensor
