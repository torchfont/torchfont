from collections.abc import Callable
from typing import Literal

import pytest
import torch

from torchfont import ElementType
from torchfont.nn.functional import outline_loss


@pytest.fixture(params=["l1", "l2", "smooth_l1"])
def loss_kind(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture
def coordinate_loss(
    loss_kind: Literal["l1", "l2", "smooth_l1"],
) -> Callable[..., torch.Tensor]:
    def loss(
        prediction: torch.Tensor,
        target_types: torch.Tensor,
        target_coords: torch.Tensor,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> torch.Tensor:
        return outline_loss(
            torch.zeros(*target_types.shape, len(ElementType)),
            prediction,
            target_types,
            target_coords,
            coordinate_loss=loss_kind,
            type_weight=0.0,
            coordinate_weight=1.0,
            reduction=reduction,
        )

    return loss


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


def test_coordinate_loss_masks_inactive_coordinates(
    coordinate_loss: Callable[..., torch.Tensor],
    loss_kind: Literal["l1", "l2", "smooth_l1"],
) -> None:
    types = _types()
    prediction = torch.zeros((len(types), 6), requires_grad=True)
    target = torch.ones_like(prediction)

    loss = coordinate_loss(prediction, types, target, reduction="none")

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
    error = 0.5 if loss_kind == "smooth_l1" else 1.0
    assert torch.equal(loss, expected.sum() * error)
    loss.backward()
    gradient = -2.0 if loss_kind == "l2" else -1.0
    torch.testing.assert_close(prediction.grad, expected * gradient)


def test_coordinate_loss_ignores_nonfinite_inactive_coordinates(
    coordinate_loss: Callable[..., torch.Tensor],
    loss_kind: Literal["l1", "l2", "smooth_l1"],
) -> None:
    types = torch.tensor([ElementType.MOVE_TO, ElementType.PAD])
    prediction = torch.zeros((2, 6))
    target = torch.tensor(
        [
            [torch.nan, torch.inf, -torch.inf, torch.nan, 1.0, 1.0],
            [torch.nan, torch.inf, -torch.inf, torch.nan, torch.nan, torch.inf],
        ]
    )

    loss = coordinate_loss(prediction, types, target, reduction="none")

    assert torch.equal(
        loss,
        torch.tensor(2.0) * (0.5 if loss_kind == "smooth_l1" else 1.0),
    )


def test_coordinate_loss_mean_is_zero_without_active_coordinates(
    coordinate_loss: Callable[..., torch.Tensor],
) -> None:
    types = torch.tensor([ElementType.CLOSE, ElementType.END, ElementType.PAD])
    prediction = torch.zeros((3, 6), requires_grad=True)

    loss = coordinate_loss(prediction, types, torch.ones_like(prediction))
    loss.backward()

    assert loss.item() == 0.0
    assert prediction.grad is not None
    assert torch.count_nonzero(prediction.grad) == 0


def test_coordinate_loss_reductions_count_only_active_coordinates(
    coordinate_loss: Callable[..., torch.Tensor],
    loss_kind: Literal["l1", "l2", "smooth_l1"],
) -> None:
    types = _types()
    prediction = torch.zeros((len(types), 6))
    target = torch.ones_like(prediction)

    error = 0.5 if loss_kind == "smooth_l1" else 1.0
    assert coordinate_loss(prediction, types, target).item() == error
    assert (
        coordinate_loss(prediction, types, target, reduction="sum").item() == 14 * error
    )


def test_coordinate_loss_rejects_prediction_shape_mismatch() -> None:
    target_types = torch.zeros(3, dtype=torch.long)
    target_coords = torch.zeros((3, 6))

    with pytest.raises(ValueError, match="same shape as the target coordinates"):
        outline_loss(
            torch.zeros((3, len(ElementType))),
            torch.zeros((2, 6)),
            target_types,
            target_coords,
        )


def test_coordinate_loss_rejects_misaligned_target_tensors() -> None:
    with pytest.raises(ValueError, match="coords must have shape"):
        outline_loss(
            torch.zeros((1, len(ElementType))),
            torch.zeros((2, 6)),
            torch.zeros(1, dtype=torch.long),
            torch.zeros((2, 6)),
        )
