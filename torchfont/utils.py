"""Utility helpers for building training pipelines with TorchFont datasets.

Examples:
    Use the built-in collate function with a :class:`~torch.utils.data.DataLoader`::

        from torch.utils.data import DataLoader
        from torchfont.datasets import GlyphDataset
        from torchfont.utils import collate_fn

        dataset = GlyphDataset(root="~/fonts")
        loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence

from torchfont.batch import GlyphBatch
from torchfont.sample import GlyphSample


def collate_fn(
    batch: Sequence[GlyphSample],
) -> GlyphBatch:
    """Collate a list of glyph samples into a padded glyph batch.

    Pads variable-length ``types`` and ``coords`` sequences to the length of
    the longest sample in the batch. Suitable for use as the ``collate_fn``
    argument of :class:`~torch.utils.data.DataLoader`.

    Args:
        batch: Sequence of :class:`~torchfont.sample.GlyphSample` values as
            returned by a TorchFont dataset.

    Returns:
        GlyphBatch: Structured batch containing padded tensors plus a validity
            mask for non-padding positions.

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
        [sample.types.shape[0] for sample in batch],
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


def collate_tuples(
    batch: Sequence[GlyphSample],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Legacy tuple-return wrapper around :func:`collate_fn`."""
    glyph_batch = collate_fn(batch)
    return (
        glyph_batch.types,
        glyph_batch.coords,
        glyph_batch.style_idx,
        glyph_batch.content_idx,
    )
