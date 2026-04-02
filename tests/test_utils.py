import pytest
import torch
from torch.utils.data import DataLoader

import torchfont.utils as utils_module
from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import Patchify
from torchfont.utils import GlyphBatch, collate_fn


def test_utils_public_api_is_batching_centered() -> None:
    assert utils_module.__all__ == ["GlyphBatch", "collate_fn"]
    assert utils_module.GlyphBatch is GlyphBatch


def test_collate_fn_basic() -> None:
    """collate_fn returns correctly shaped and typed tensors."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert len(dataset) >= 2
    batch = [dataset[i] for i in range(2)]
    glyph_batch = collate_fn(batch)

    assert isinstance(glyph_batch, GlyphBatch)
    assert glyph_batch.types.dtype == torch.long
    assert glyph_batch.types.ndim == 2
    assert glyph_batch.coords.dtype == torch.float32
    assert glyph_batch.coords.ndim == 3
    assert glyph_batch.coords.shape[2] == 6
    assert glyph_batch.style_idx.dtype == torch.long
    assert glyph_batch.style_idx.shape == (2,)
    assert glyph_batch.content_idx.dtype == torch.long
    assert glyph_batch.content_idx.shape == (2,)
    assert glyph_batch.mask.dtype == torch.bool
    assert glyph_batch.mask.shape == glyph_batch.types.shape


def test_collate_fn_with_dataloader() -> None:
    """collate_fn works as the collate_fn argument of DataLoader."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    loader = DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
    batch = next(iter(loader))

    assert isinstance(batch, GlyphBatch)
    assert batch.types.ndim == 2
    assert batch.coords.ndim == 3
    assert batch.style_idx.ndim == 1
    assert batch.content_idx.ndim == 1
    assert batch.mask.ndim == 2


def test_collate_fn_pads_to_longest() -> None:
    """collate_fn zero-pads shorter sequences to match the longest in batch."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    batch = [dataset[i] for i in range(len(dataset))]
    glyph_batch = collate_fn(batch)
    types_t = glyph_batch.types
    coords_t = glyph_batch.coords

    lengths = [sample.types.shape[0] for sample in batch]
    max_len = max(lengths)
    assert types_t.shape[1] == max_len
    assert coords_t.shape[1] == max_len

    # Verify that positions beyond each sample's original length are zero-padded.
    for b, orig_len in enumerate(lengths):
        if orig_len < max_len:
            padded_types = types_t[b, orig_len:]
            padded_coords = coords_t[b, orig_len:, :]
            assert torch.all(padded_types == 0)
            assert torch.all(padded_coords == 0.0)
            assert not torch.any(glyph_batch.mask[b, orig_len:])


def test_collate_fn_keeps_label_tensors_on_sample_device() -> None:
    """collate_fn keeps indices and mask on the same device as sample tensors."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    batch = [dataset[i] for i in range(2)]
    glyph_batch = collate_fn(batch)

    assert glyph_batch.style_idx.device == glyph_batch.types.device
    assert glyph_batch.content_idx.device == glyph_batch.types.device
    assert glyph_batch.mask.device == glyph_batch.types.device


def test_collate_fn_preserves_trailing_patch_dimensions() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x45),
        transform=Patchify(4),
    )

    batch = [dataset[i] for i in range(2)]
    glyph_batch = collate_fn(batch)

    assert glyph_batch.types.ndim == 3
    assert glyph_batch.coords.ndim == 4
    assert glyph_batch.types.shape[:2] == glyph_batch.mask.shape
    assert glyph_batch.types.shape[2] == 4
    assert glyph_batch.coords.shape[2:] == (4, 6)


def test_collate_fn_rejects_empty_batch() -> None:
    with pytest.raises(ValueError, match="batch must be non-empty"):
        collate_fn([])


def test_collate_fn_rejects_incompatible_trailing_types_shapes() -> None:
    samples = [
        GlyphSample(
            types=torch.tensor([1, 2], dtype=torch.long),
            coords=torch.zeros(2, 6),
            style_idx=0,
            content_idx=0,
        ),
        GlyphSample(
            types=torch.tensor([[1, 2]], dtype=torch.long),
            coords=torch.zeros(1, 6),
            style_idx=1,
            content_idx=1,
        ),
    ]

    with pytest.raises(ValueError, match="trailing types shape"):
        collate_fn(samples)


def test_collate_fn_rejects_incompatible_trailing_coords_shapes() -> None:
    samples = [
        GlyphSample(
            types=torch.tensor([1, 2], dtype=torch.long),
            coords=torch.zeros(2, 6),
            style_idx=0,
            content_idx=0,
        ),
        GlyphSample(
            types=torch.tensor([1, 2, 3], dtype=torch.long),
            coords=torch.zeros(3, 1, 6),
            style_idx=1,
            content_idx=1,
        ),
    ]

    with pytest.raises(ValueError, match="trailing coords shape"):
        collate_fn(samples)


def test_collate_fn_rejects_zero_dim_trailing_types_inputs() -> None:
    samples = [
        GlyphSample(
            types=torch.tensor(1, dtype=torch.long),
            coords=torch.zeros(1, 6),
            style_idx=0,
            content_idx=0,
        ),
    ]

    with pytest.raises(ValueError, match="at least 1-D for 'types'"):
        collate_fn(samples)


def test_collate_fn_rejects_zero_dim_trailing_coords_inputs() -> None:
    samples = [
        GlyphSample(
            types=torch.tensor([1], dtype=torch.long),
            coords=torch.tensor(0.0),
            style_idx=0,
            content_idx=0,
        ),
    ]

    with pytest.raises(ValueError, match="at least 1-D for 'coords'"):
        collate_fn(samples)
