import torch
from torch.utils.data import DataLoader

from torchfont.datasets import FontFolder
from torchfont.utils import collate_fn


def test_collate_fn_basic() -> None:
    """collate_fn returns correctly shaped and typed tensors."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    assert len(dataset) >= 2
    batch = [dataset[i] for i in range(2)]
    types_t, coords_t, style_t, content_t = collate_fn(batch)

    assert types_t.dtype == torch.long
    assert types_t.ndim == 2
    assert coords_t.dtype == torch.float32
    assert coords_t.ndim == 3
    assert coords_t.shape[2] == 6
    assert style_t.dtype == torch.long
    assert style_t.shape == (2,)
    assert content_t.dtype == torch.long
    assert content_t.shape == (2,)


def test_collate_fn_with_dataloader() -> None:
    """collate_fn works as the collate_fn argument of DataLoader."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))
    types_t, coords_t, style_t, content_t = batch

    assert types_t.ndim == 2
    assert coords_t.ndim == 3
    assert style_t.ndim == 1
    assert content_t.ndim == 1


def test_collate_fn_pads_to_longest() -> None:
    """collate_fn zero-pads shorter sequences to match the longest in batch."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
    )

    batch = [dataset[i] for i in range(len(dataset))]
    types_t, coords_t, _, _ = collate_fn(batch)

    max_len = max(types.shape[0] for types, _, _, _ in batch)
    assert types_t.shape[1] == max_len
    assert coords_t.shape[1] == max_len
