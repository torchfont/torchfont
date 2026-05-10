"""Utility helpers for building training pipelines with TorchFont datasets.

Examples:
    Use the built-in collate function with a :class:`~torch.utils.data.DataLoader`::

        from torch.utils.data import DataLoader
        from torchfont.datasets import GlyphDataset
        from torchfont.utils import collate_outline

        dataset = GlyphDataset(root="~/fonts", transform=lambda s: (s.types, s.coords))
        loader = DataLoader(dataset, batch_size=64, collate_fn=collate_outline)

"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch.nn.utils.rnn import pad_sequence

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


def collate_outline(
    batch: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    """Collate a list of ``(types, coords)`` pairs into padded batch tensors.

    Pads the leading variable-length sequence dimension of ``types`` and
    ``coords`` to the longest sample in the batch. Suitable for use as the
    ``collate_outline`` argument of :class:`~torch.utils.data.DataLoader`.

    Args:
        batch: Sequence of ``(types, coords)`` pairs as returned by a dataset
            transform.

    Returns:
        Tuple ``(types, coords)`` where each tensor has a new leading batch
        dimension. Any trailing dimensions are preserved.

    Examples:
        Plug directly into a DataLoader::

            from torch.utils.data import DataLoader
            from torchfont.utils import collate_outline

            loader = DataLoader(dataset, batch_size=32, collate_fn=collate_outline)

    """
    if not batch:
        msg = "batch must be non-empty"
        raise ValueError(msg)

    types_list = [item[0] for item in batch]
    coords_list = [item[1] for item in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    return types_tensor, coords_tensor


__all__ = ["collate_outline"]
