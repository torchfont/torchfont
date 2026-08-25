"""``torchfont.nn`` operating on single and batched outline tensors."""

from __future__ import annotations

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from torchfont import ElementType, Outline
from torchfont.nn import OutlineEmbedding, OutlineLoss
from torchfont.nn import functional as F  # noqa: N812


def _outline(length: int = 4) -> Outline:
    types = torch.tensor(
        [ElementType.MOVE_TO, ElementType.CURVE_TO, ElementType.CLOSE, ElementType.END],
        dtype=torch.long,
    )[:length]
    return Outline(types, torch.rand(length, 6))


@pytest.fixture
def batch() -> tuple[torch.Tensor, torch.Tensor]:
    outlines = [_outline(), _outline(3)]
    return (
        pad_sequence(
            [outline.types for outline in outlines],
            batch_first=True,
            padding_value=ElementType.PAD,
        ),
        pad_sequence([outline.coords for outline in outlines], batch_first=True),
    )


def test_embedding_accepts_single_outline_tensors() -> None:
    outline = _outline()
    assert OutlineEmbedding(8)(outline.types, outline.coords).shape == (4, 8)


def test_embedding_accepts_batched_outline_tensors(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    assert OutlineEmbedding(8)(*batch).shape == (2, 4, 8)


def test_loss_accepts_batched_outline_tensors(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = batch
    logits = torch.zeros(*types.shape, len(ElementType))

    assert OutlineLoss()(logits, torch.zeros_like(coords), types, coords).ndim == 0


def test_coordinate_mse_loss_accepts_batched_outline_tensors(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = batch

    assert F.coordinate_mse_loss(torch.zeros_like(coords), types, coords).ndim == 0


def test_loss_ignores_padding_introduced_by_pad_sequence() -> None:
    single = _outline(3)
    types = pad_sequence(
        [single.types, single.types],
        batch_first=True,
        padding_value=ElementType.PAD,
    )
    coords = pad_sequence([single.coords, single.coords], batch_first=True)
    padded_loss = OutlineLoss()(
        torch.zeros(*types.shape, len(ElementType)),
        torch.zeros_like(coords),
        types,
        coords,
    )
    unpadded_loss = OutlineLoss()(
        torch.zeros(1, *single.shape, len(ElementType)),
        torch.zeros(1, *single.coords.shape),
        single.types.unsqueeze(0),
        single.coords.unsqueeze(0),
    )

    assert torch.allclose(padded_loss, unpadded_loss)
