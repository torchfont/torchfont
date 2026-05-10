import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

import torchfont.utils as utils_module
from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import patchify
from torchfont.utils import collate_outline


def _to_pair(sample: GlyphSample) -> tuple[Tensor, Tensor]:
    return sample.types, sample.coords


def test_utils_public_api_is_batching_centered() -> None:
    assert utils_module.__all__ == ["collate_outline"]
    assert utils_module.collate_outline is collate_outline


def test_collate_outline_basic() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
        transform=_to_pair,
    )

    assert len(dataset) >= 2
    batch = [dataset[i] for i in range(2)]
    types_t, coords_t = collate_outline(batch)

    assert types_t.dtype == torch.long
    assert types_t.ndim == 2
    assert coords_t.dtype == torch.float32
    assert coords_t.ndim == 3
    assert coords_t.shape[2] == 6


def test_collate_outline_with_dataloader() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
        transform=_to_pair,
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_outline)
    types_t, coords_t = next(iter(loader))

    assert types_t.ndim == 2
    assert coords_t.ndim == 3


def test_collate_outline_pads_to_longest() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
        transform=_to_pair,
    )

    batch = [dataset[i] for i in range(len(dataset))]
    types_t, coords_t = collate_outline(batch)

    lengths = [item[0].shape[0] for item in batch]
    max_len = max(lengths)
    assert types_t.shape[1] == max_len
    assert coords_t.shape[1] == max_len

    for b, orig_len in enumerate(lengths):
        if orig_len < max_len:
            assert torch.all(types_t[b, orig_len:] == 0)
            assert torch.all(coords_t[b, orig_len:, :] == 0.0)


def test_collate_outline_keeps_tensors_on_same_device() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
        transform=_to_pair,
    )

    batch = [dataset[i] for i in range(2)]
    types_t, coords_t = collate_outline(batch)

    assert types_t.device == coords_t.device


def test_collate_outline_preserves_trailing_patch_dimensions() -> None:
    def patchify_transform(sample: GlyphSample) -> tuple[Tensor, Tensor]:
        return patchify(sample.types, sample.coords, patch_size=4)

    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x45),
        transform=patchify_transform,
    )

    batch = [dataset[i] for i in range(2)]
    types_t, coords_t = collate_outline(batch)

    assert types_t.ndim == 3
    assert coords_t.ndim == 4
    assert types_t.shape[2] == 4
    assert coords_t.shape[2:] == (4, 6)


def test_collate_outline_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch must be non-empty"):
        collate_outline([])
