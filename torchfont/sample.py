"""Structured representation of a single glyph sample."""

from typing import NamedTuple

from torch import Tensor


class GlyphSample(NamedTuple):
    """One glyph sample returned by a dataset.

    Using a NamedTuple means fields can be accessed by name rather than by
    position, so downstream code does not depend on tuple ordering.
    PyTorch's default DataLoader collation handles NamedTuples natively.

    Attributes:
        types (Tensor): 1-D long tensor of pen command types.
        coords (Tensor): 2-D float tensor of shape ``(N, 6)`` holding the
            coordinate data for each command.
        style_idx (int): Index into the dataset's ``style_classes`` list.
        content_idx (int): Index into the dataset's ``content_classes`` list.

    Examples:
        Access fields by name rather than by position::

            sample = dataset[0]
            print(sample.types.shape, sample.style_idx)

    """

    types: Tensor
    coords: Tensor
    style_idx: int
    content_idx: int


__all__ = ["GlyphSample"]
