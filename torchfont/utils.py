"""Utility helpers for building training pipelines with TorchFont datasets.

Examples:
    Use the built-in collate function with a :class:`~torch.utils.data.DataLoader`::

        from torch.utils.data import DataLoader
        from torchfont.datasets import FontFolder
        from torchfont.utils import collate_fn

        dataset = FontFolder(root="~/fonts")
        loader = DataLoader(dataset, batch_size=64, collate_fn=collate_fn)

"""

from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence


def collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Collate a list of glyph samples into padded batch tensors.

    Pads variable-length ``types`` and ``coords`` sequences to the length of
    the longest sample in the batch. Suitable for use as the ``collate_fn``
    argument of :class:`~torch.utils.data.DataLoader`.

    Args:
        batch: Sequence of ``(types, coords, style_idx, content_idx)`` tuples
            as returned by a :class:`~torchfont.datasets.FontFolder` dataset.

    Returns:
        tuple[Tensor, Tensor, Tensor, Tensor]:
            - ``types`` — ``(B, L)`` long tensor of command-type codes,
              zero-padded.
            - ``coords`` — ``(B, L, 6)`` float tensor of coordinate data,
              zero-padded.
            - ``style_labels`` — ``(B,)`` long tensor of style class indices.
            - ``content_labels`` — ``(B,)`` long tensor of content class
              indices.

    Examples:
        Plug directly into a DataLoader::

            from torch.utils.data import DataLoader
            from torchfont.utils import collate_fn

            loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

    """
    types_list = [types for types, _, _, _ in batch]
    coords_list = [coords for _, coords, _, _ in batch]
    style_label_list = [style for _, _, style, _ in batch]
    content_label_list = [content for _, _, _, content in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    style_label_tensor = torch.as_tensor(style_label_list, dtype=torch.long)
    content_label_tensor = torch.as_tensor(content_label_list, dtype=torch.long)

    return types_tensor, coords_tensor, style_label_tensor, content_label_tensor
