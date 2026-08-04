import pytest
import torch

from torchfont import ElementType
from torchfont.nn.functional import coordinate_loss


def _types() -> torch.Tensor:
    return torch.tensor(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.QUAD_TO,
            ElementType.CURVE_TO,
            ElementType.CLOSE,
            ElementType.END,
            ElementType.PAD,
        ]
    )


def test_coordinate_loss_masks_inactive_coordinates() -> None:
    types = _types()
    prediction = torch.zeros((len(types), 6))
    target = torch.ones_like(prediction)

    loss = coordinate_loss(prediction, target, types, reduction="none")

    expected = torch.tensor(
        [
            [0, 0, 0, 0, 1, 1],
            [0, 0, 0, 0, 1, 1],
            [1, 1, 0, 0, 1, 1],
            [1, 1, 1, 1, 1, 1],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(loss, expected)


def test_coordinate_loss_ignores_nonfinite_inactive_coordinates() -> None:
    types = torch.tensor([ElementType.MOVE_TO, ElementType.PAD])
    prediction = torch.zeros((2, 6))
    target = torch.tensor(
        [
            [torch.nan, torch.inf, -torch.inf, torch.nan, 1.0, 1.0],
            [torch.nan, torch.inf, -torch.inf, torch.nan, torch.nan, torch.inf],
        ]
    )

    loss = coordinate_loss(prediction, target, types, reduction="none")

    assert torch.equal(
        loss,
        torch.tensor([[0.0, 0.0, 0.0, 0.0, 1.0, 1.0], [0.0] * 6]),
    )


def test_coordinate_loss_mean_is_zero_without_active_coordinates() -> None:
    types = torch.tensor([ElementType.CLOSE, ElementType.END, ElementType.PAD])
    prediction = torch.zeros((3, 6), requires_grad=True)

    loss = coordinate_loss(prediction, torch.ones_like(prediction), types)
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_coordinate_loss_reductions_count_only_active_coordinates() -> None:
    types = _types()
    prediction = torch.zeros((len(types), 6))
    target = torch.ones_like(prediction)

    assert coordinate_loss(prediction, target, types).item() == 1.0
    assert coordinate_loss(prediction, target, types, reduction="sum").item() == 14.0


def test_coordinate_loss_backpropagates_only_through_active_coordinates() -> None:
    types = torch.tensor([ElementType.QUAD_TO, ElementType.PAD])
    prediction = torch.zeros((2, 6), requires_grad=True)
    target = torch.ones_like(prediction)

    coordinate_loss(prediction, target, types, reduction="sum").backward()

    grad = prediction.grad
    assert grad is not None
    assert torch.equal(grad[0], torch.tensor([-2, -2, 0, 0, -2, -2.0]))
    assert torch.count_nonzero(grad[1]) == 0


@pytest.mark.parametrize(
    ("prediction_shape", "target_shape", "types_shape", "match"),
    [
        ((2, 6), (3, 6), (2,), "same shape"),
        ((2, 5), (2, 5), (2,), "must have shape"),
        ((2, 6), (2, 6), (1, 2), "types shape must match"),
    ],
)
def test_coordinate_loss_rejects_misaligned_shapes(
    prediction_shape: tuple[int, ...],
    target_shape: tuple[int, ...],
    types_shape: tuple[int, ...],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        coordinate_loss(
            torch.zeros(prediction_shape),
            torch.zeros(target_shape),
            torch.zeros(types_shape, dtype=torch.long),
        )


def test_coordinate_loss_rejects_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="not a valid value"):
        coordinate_loss(
            torch.zeros((1, 6)),
            torch.zeros((1, 6)),
            torch.tensor([ElementType.MOVE_TO]),
            reduction="invalid",  # ty: ignore[invalid-argument-type]
        )
