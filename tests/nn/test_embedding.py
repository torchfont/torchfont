import pytest
import torch

from torchfont import COORD_DIM, TYPE_DIM, ElementType, Outline
from torchfont.nn import OutlineEmbedding


def test_outline_embedding_combines_types_and_coordinates() -> None:
    embedding = OutlineEmbedding(3)
    with torch.no_grad():
        embedding.type_embedding.weight.fill_(1.0)
        embedding.coord_projection.weight.fill_(2.0)
    outline = Outline(
        torch.tensor([[ElementType.MOVE_TO, ElementType.LINE_TO]]),
        torch.ones((1, 2, 6)),
    )

    output = embedding(outline)

    assert output.shape == (1, 2, 3)
    assert torch.equal(output, torch.full((1, 2, 3), 5.0))


def test_outline_embedding_ignores_inactive_coordinates() -> None:
    embedding = OutlineEmbedding(1)
    with torch.no_grad():
        embedding.type_embedding.weight.zero_()
        embedding.coord_projection.weight.fill_(1.0)
    outline = Outline(
        torch.tensor([ElementType.MOVE_TO, ElementType.QUAD_TO]),
        torch.tensor(
            [
                [100.0, 100.0, 100.0, 100.0, 1.0, 2.0],
                [1.0, 2.0, 100.0, 100.0, 3.0, 4.0],
            ]
        ),
    )

    output = embedding(outline)

    assert torch.equal(output[:, 0], torch.tensor([3.0, 10.0]))


def test_outline_embedding_ignores_nonfinite_inactive_coordinates() -> None:
    embedding = OutlineEmbedding(1)
    with torch.no_grad():
        embedding.type_embedding.weight.zero_()
        embedding.coord_projection.weight.fill_(1.0)
    outline = Outline(
        torch.tensor([ElementType.MOVE_TO, ElementType.PAD]),
        torch.tensor(
            [
                [torch.nan, torch.inf, -torch.inf, torch.nan, 1.0, 2.0],
                [torch.nan, torch.inf, -torch.inf, torch.nan, torch.nan, torch.inf],
            ]
        ),
    )

    output = embedding(outline)

    assert torch.equal(output, torch.tensor([[3.0], [0.0]]))


def test_outline_embedding_zeroes_padding_tokens() -> None:
    embedding = OutlineEmbedding(4)
    outline = Outline(
        torch.tensor([ElementType.MOVE_TO, ElementType.PAD]), torch.ones((2, 6))
    )

    output = embedding(outline)

    assert torch.count_nonzero(output[0]) > 0
    assert torch.count_nonzero(output[1]) == 0


def test_outline_embedding_supports_factory_dtype() -> None:
    embedding = OutlineEmbedding(4, dtype=torch.float64)
    outline = Outline(
        torch.tensor([ElementType.MOVE_TO]), torch.ones((1, 6), dtype=torch.float64)
    )

    output = embedding(outline)

    assert output.dtype == torch.float64
    assert embedding.type_embedding.weight.dtype == torch.float64
    assert embedding.coord_projection.weight.dtype == torch.float64


def test_outline_embedding_registers_parameters() -> None:
    embedding = OutlineEmbedding(4)

    assert set(embedding.state_dict()) == {
        "type_embedding.weight",
        "coord_projection.weight",
    }
    assert repr(embedding).startswith("OutlineEmbedding(\n  embedding_dim=4")


def test_outline_embedding_initializes_both_branches_from_one_bound() -> None:
    embedding = OutlineEmbedding(1024)
    bound = (TYPE_DIM + COORD_DIM) ** -0.5
    expected_std = bound / 3**0.5
    type_weight = embedding.type_embedding.weight.detach()[1:]
    coord_weight = embedding.coord_projection.weight.detach()

    assert type_weight.abs().max() <= bound
    assert coord_weight.abs().max() <= bound
    assert float(type_weight.std()) == pytest.approx(expected_std, rel=0.1)
    assert float(coord_weight.std()) == pytest.approx(expected_std, rel=0.1)
    assert torch.count_nonzero(embedding.type_embedding.weight[0]) == 0


def test_outline_embedding_reset_parameters_redraws_and_keeps_padding_zero() -> None:
    embedding = OutlineEmbedding(128)
    before = embedding.type_embedding.weight.clone()

    embedding.reset_parameters()

    assert not torch.equal(embedding.type_embedding.weight[1:], before[1:])
    assert torch.count_nonzero(embedding.type_embedding.weight[0]) == 0
