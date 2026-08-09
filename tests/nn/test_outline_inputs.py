"""``torchfont.nn`` operating on single and batched :class:`torchfont.Outline`."""

from __future__ import annotations

import pytest
import torch

from torchfont import COORD_DIM, TYPE_DIM, ElementType, Outline, pad_outlines
from torchfont.nn import OutlineEmbedding, OutlineLoss
from torchfont.nn import functional as F  # noqa: N812


def _outline(length: int = 4) -> Outline:
    types = torch.tensor(
        [ElementType.MOVE_TO, ElementType.CURVE_TO, ElementType.CLOSE, ElementType.END],
        dtype=torch.long,
    )[:length]
    return Outline(types, torch.rand(length, COORD_DIM))


@pytest.fixture
def batch() -> Outline:
    return pad_outlines([_outline(), _outline(3)])


def test_embedding_accepts_a_single_outline() -> None:
    assert OutlineEmbedding(8)(_outline()).shape == (4, 8)


def test_embedding_accepts_a_batched_outline(batch: Outline) -> None:
    assert OutlineEmbedding(8)(batch).shape == (2, 4, 8)


def test_loss_accepts_a_batched_outline(batch: Outline) -> None:
    logits = torch.zeros(*batch.shape, TYPE_DIM)
    prediction = torch.zeros(*batch.shape, COORD_DIM)

    assert OutlineLoss()(logits, prediction, batch).ndim == 0


def test_coordinate_loss_accepts_a_batched_outline(batch: Outline) -> None:
    prediction = torch.zeros(*batch.shape, COORD_DIM)

    assert F.coordinate_loss(prediction, batch).ndim == 0


def test_loss_ignores_padding_introduced_by_pad_outlines() -> None:
    single = _outline(3)
    batch = pad_outlines([single, single])
    padded_loss = OutlineLoss()(
        torch.zeros(*batch.shape, TYPE_DIM),
        torch.zeros(*batch.shape, COORD_DIM),
        batch,
    )

    unpadded = pad_outlines([single])
    unpadded_loss = OutlineLoss()(
        torch.zeros(*unpadded.shape, TYPE_DIM),
        torch.zeros(*unpadded.shape, COORD_DIM),
        unpadded,
    )

    assert torch.allclose(padded_loss, unpadded_loss)
