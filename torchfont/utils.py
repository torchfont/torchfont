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
        targets (Tensor): Long tensor of shape ``(B, 2)`` where column 0 is
            style indices and column 1 is content indices.
        metrics (Tensor): Float tensor of shape ``(B, 15)`` holding per-sample
            metrics in the same column order as ``GlyphSample.metrics``.

    Note:
        ``glyph_name`` is not included because it cannot be collated into a
        tensor.

    """

    types: Tensor
    coords: Tensor
    targets: Tensor
    metrics: Tensor


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
        GlyphBatch: Structured batch containing padded tensors. Any trailing
            dimensions produced by transforms such as ``Patchify`` are
            preserved.

    Examples:
        Plug directly into a DataLoader::

            from torch.utils.data import DataLoader
            from torchfont.utils import collate_fn

            loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    """
    if not batch:
        msg = "batch must be non-empty"
        raise ValueError(msg)

    types_list = [sample.types for sample in batch]
    coords_list = [sample.coords for sample in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    targets_tensor = torch.tensor(
        [(s.style_idx, s.content_idx) for s in batch], dtype=torch.long
    )
    metrics_tensor = torch.frombuffer(
        bytearray(b"".join(s.metrics for s in batch)),
        dtype=torch.float32,
    ).view(len(batch), 15)

    return GlyphBatch(
        types=types_tensor,
        coords=coords_tensor,
        targets=targets_tensor,
        metrics=metrics_tensor,
    )


__all__ = ["GlyphBatch", "collate_fn"]
