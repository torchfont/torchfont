from __future__ import annotations

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence

from torchfont import ElementType, Outline
from torchfont.nn import OutlineEmbedding, OutlineLoss


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


def test_embedding_preserves_each_outline_in_a_padded_batch(
    batch: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = batch
    embedding = OutlineEmbedding(8)
    output = embedding(types, coords)

    assert output.shape == (2, 4, 8)
    for index, length in enumerate((4, 3)):
        torch.testing.assert_close(
            output[index, :length],
            embedding(types[index, :length], coords[index, :length]),
        )
    assert torch.count_nonzero(output[1, 3]) == 0


def test_loss_ignores_padding_introduced_by_pad_sequence() -> None:
    outlines = [_outline(), _outline(3)]
    types = pad_sequence(
        [outline.types for outline in outlines],
        batch_first=True,
        padding_value=ElementType.PAD,
    )
    coords = pad_sequence([outline.coords for outline in outlines], batch_first=True)
    assert types[1, -1] == ElementType.PAD
    padded_loss = OutlineLoss()(
        torch.zeros(*types.shape, len(ElementType)),
        torch.zeros_like(coords),
        types,
        coords,
    )
    unpadded_types = torch.cat([outline.types for outline in outlines])
    unpadded_coords = torch.cat([outline.coords for outline in outlines])
    unpadded_loss = OutlineLoss()(
        torch.zeros(len(unpadded_types), len(ElementType)),
        torch.zeros_like(unpadded_coords),
        unpadded_types,
        unpadded_coords,
    )

    assert torch.allclose(padded_loss, unpadded_loss)
