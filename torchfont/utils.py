"""Utility helpers for building training pipelines with TorchFont datasets.

Examples:
    Use the built-in collate function with a :class:`~torch.utils.data.DataLoader`::

        from torch.utils.data import DataLoader
        from torchfont.datasets import GlyphDataset
        from torchfont.utils import collate_fn

        dataset = GlyphDataset(root="~/fonts")
        loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torchfont.datasets import GlyphSample


class GlyphBatch(NamedTuple):
    """One collated glyph batch.

    Attributes:
        types (Tensor): Long tensor of shape ``(B, L, ...)`` holding padded
            command values. Only the leading sequence dimension ``L`` is padded;
            any trailing dimensions are preserved.
        coords (Tensor): Float tensor of shape ``(B, L, ...)`` holding padded
            coordinate values. Only the leading sequence dimension ``L`` is
            padded; trailing dimensions such as ``patch_size`` are preserved.
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


def _validate_trailing_shape(name: str, tensors: Sequence[Tensor]) -> None:
    """Ensure every tensor agrees past the leading sequence dimension."""
    if not tensors:
        return

    for idx, tensor in enumerate(tensors):
        if tensor.ndim < 1:
            msg = (
                f"all samples must be at least 1-D for '{name}'; "
                f"found 0-D tensor at batch index {idx}"
            )
            raise ValueError(msg)

    expected = tuple(tensors[0].shape[1:])
    for idx, tensor in enumerate(tensors[1:], start=1):
        actual = tuple(tensor.shape[1:])
        if actual != expected:
            msg = (
                f"all samples must share the same trailing {name} shape; "
                f"expected {expected}, got {actual} at batch index {idx}"
            )
            raise ValueError(msg)


def collate_fn(
    batch: Sequence[GlyphSample],
) -> GlyphBatch:
    """Collate a list of glyph samples into a padded glyph batch.

    Pads the leading variable-length sequence dimension of ``types`` and
    ``coords`` to the longest sample in the batch. Suitable for use as the
    ``collate_fn`` argument of :class:`~torch.utils.data.DataLoader`.

    Args:
        batch: Sequence of :class:`~torchfont.datasets.GlyphSample` values as
            returned by a TorchFont dataset.

    Returns:
        GlyphBatch: Structured batch containing padded tensors plus a validity
            mask for non-padding positions. Any trailing dimensions produced by
            transforms such as ``Patchify`` are preserved.

    Examples:
        Plug directly into a DataLoader::

            from torch.utils.data import DataLoader
            from torchfont.utils import collate_fn

            loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    """
    types_list = [sample.types for sample in batch]
    coords_list = [sample.coords for sample in batch]
    style_label_list = [sample.style_idx for sample in batch]
    content_label_list = [sample.content_idx for sample in batch]

    _validate_trailing_shape("types", types_list)
    _validate_trailing_shape("coords", coords_list)

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    style_label_tensor = torch.as_tensor(
        style_label_list,
        dtype=torch.long,
        device=types_tensor.device,
    )
    content_label_tensor = torch.as_tensor(
        content_label_list,
        dtype=torch.long,
        device=types_tensor.device,
    )
    lengths = torch.as_tensor(
        [t.shape[0] for t in types_list],
        dtype=torch.long,
        device=types_tensor.device,
    )
    steps = torch.arange(types_tensor.shape[1], device=types_tensor.device)
    mask = steps.unsqueeze(0) < lengths.unsqueeze(1)

    return GlyphBatch(
        types=types_tensor,
        coords=coords_tensor,
        style_idx=style_label_tensor,
        content_idx=content_label_tensor,
        mask=mask,
    )


__all__ = ["GlyphBatch", "collate_fn"]
