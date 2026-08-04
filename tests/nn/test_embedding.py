import pytest
import torch

from torchfont import ElementType
from torchfont.nn import OutlineEmbedding


def test_outline_embedding_combines_types_and_coordinates() -> None:
    embedding = OutlineEmbedding(3)
    with torch.no_grad():
        embedding.type_embedding.weight.fill_(1.0)
        embedding.coord_projection.weight.fill_(2.0)
    types = torch.tensor([[ElementType.MOVE_TO, ElementType.LINE_TO]])
    coords = torch.ones((1, 2, 6))

    output = embedding(types, coords)

    assert output.shape == (1, 2, 3)
    assert torch.equal(output, torch.full((1, 2, 3), 5.0))


def test_outline_embedding_ignores_inactive_coordinates() -> None:
    embedding = OutlineEmbedding(1)
    with torch.no_grad():
        embedding.type_embedding.weight.zero_()
        embedding.coord_projection.weight.fill_(1.0)
    types = torch.tensor([ElementType.MOVE_TO, ElementType.QUAD_TO])
    coords = torch.tensor(
        [
            [100.0, 100.0, 100.0, 100.0, 1.0, 2.0],
            [1.0, 2.0, 100.0, 100.0, 3.0, 4.0],
        ]
    )

    output = embedding(types, coords)

    assert torch.equal(output[:, 0], torch.tensor([3.0, 10.0]))


def test_outline_embedding_ignores_nonfinite_inactive_coordinates() -> None:
    embedding = OutlineEmbedding(1)
    with torch.no_grad():
        embedding.type_embedding.weight.zero_()
        embedding.coord_projection.weight.fill_(1.0)
    types = torch.tensor([ElementType.MOVE_TO, ElementType.PAD])
    coords = torch.tensor(
        [
            [torch.nan, torch.inf, -torch.inf, torch.nan, 1.0, 2.0],
            [torch.nan, torch.inf, -torch.inf, torch.nan, torch.nan, torch.inf],
        ]
    )

    output = embedding(types, coords)

    assert torch.equal(output, torch.tensor([[3.0], [0.0]]))


def test_outline_embedding_zeroes_padding_tokens() -> None:
    embedding = OutlineEmbedding(4)
    types = torch.tensor([ElementType.MOVE_TO, ElementType.PAD])
    coords = torch.ones((2, 6))

    output = embedding(types, coords)

    assert torch.count_nonzero(output[0]) > 0
    assert torch.count_nonzero(output[1]) == 0


def test_outline_embedding_supports_factory_dtype() -> None:
    embedding = OutlineEmbedding(4, dtype=torch.float64)
    types = torch.tensor([ElementType.MOVE_TO])
    coords = torch.ones((1, 6), dtype=torch.float64)

    output = embedding(types, coords)

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


def test_outline_embedding_rejects_misaligned_shapes() -> None:
    embedding = OutlineEmbedding(4)

    with pytest.raises(ValueError, match="types shape must match"):
        embedding(torch.zeros(2, dtype=torch.long), torch.zeros((1, 2, 6)))
