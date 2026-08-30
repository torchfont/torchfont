import pytest
import torch
from torch.nn import functional as torch_functional

from torchfont import ElementType
from torchfont.nn import OutlineLoss
from torchfont.nn import functional as font_functional


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    type_logits = torch.zeros((3, len(ElementType)), requires_grad=True)
    coordinate_prediction = torch.zeros((3, 6), requires_grad=True)
    target_types = torch.tensor(
        [ElementType.MOVE_TO, ElementType.CURVE_TO, ElementType.PAD]
    )
    target_coords = torch.ones_like(coordinate_prediction)
    return type_logits, coordinate_prediction, target_types, target_coords


def test_outline_loss_combines_independently_averaged_losses() -> None:
    type_logits, prediction, target_types, target_coords = _inputs()

    loss = font_functional.outline_loss(
        type_logits,
        prediction,
        target_types,
        target_coords,
        type_weight=2.0,
        coordinate_weight=3.0,
    )

    expected_type_loss = torch_functional.cross_entropy(
        type_logits,
        target_types,
        ignore_index=ElementType.PAD,
    )
    expected_coord_loss = font_functional.coordinate_mse_loss(
        prediction, target_types, target_coords
    )
    assert torch.allclose(loss, 2 * expected_type_loss + 3 * expected_coord_loss)


def test_outline_loss_module_delegates_to_functional_loss() -> None:
    inputs = _inputs()
    criterion = OutlineLoss(type_weight=0.25, coordinate_weight=2.0, reduction="sum")

    actual = criterion(*inputs)
    expected = font_functional.outline_loss(
        *inputs,
        type_weight=0.25,
        coordinate_weight=2.0,
        reduction="sum",
    )

    assert torch.equal(actual, expected)
    assert repr(criterion) == (
        "OutlineLoss(type_weight=0.25, coordinate_weight=2.0, reduction='sum')"
    )


def test_outline_loss_none_returns_loss_per_outline() -> None:
    type_logits, prediction, target_types, target_coords = _inputs()
    type_logits = type_logits.unsqueeze(0).expand(2, -1, -1)
    prediction = prediction.detach().unsqueeze(0).expand(2, -1, -1)
    target_types = target_types.unsqueeze(0).expand(2, -1)
    target_coords = target_coords.unsqueeze(0).expand(2, -1, -1)

    loss = font_functional.outline_loss(
        type_logits,
        prediction,
        target_types,
        target_coords,
        type_weight=2.0,
        coordinate_weight=3.0,
        reduction="none",
    )
    expected_type_loss = torch_functional.cross_entropy(
        type_logits.transpose(1, 2),
        target_types,
        ignore_index=ElementType.PAD,
        reduction="none",
    ).sum(dim=-1)
    expected_coord_loss = font_functional.coordinate_mse_loss(
        prediction, target_types, target_coords, reduction="none"
    ).sum(dim=(-2, -1))

    assert loss.shape == (2,)
    assert torch.allclose(loss, 2 * expected_type_loss + 3 * expected_coord_loss)


def test_outline_loss_sum_sums_unreduced_outline_losses() -> None:
    type_logits, prediction, target_types, target_coords = _inputs()
    type_logits = type_logits.unsqueeze(0).expand(2, -1, -1)
    prediction = prediction.detach().unsqueeze(0).expand(2, -1, -1)
    target_types = target_types.unsqueeze(0).expand(2, -1)
    target_coords = target_coords.unsqueeze(0).expand(2, -1, -1)

    unreduced = font_functional.outline_loss(
        type_logits, prediction, target_types, target_coords, reduction="none"
    )
    reduced = font_functional.outline_loss(
        type_logits, prediction, target_types, target_coords, reduction="sum"
    )

    assert torch.equal(reduced, unreduced.sum())


def test_outline_loss_backpropagates_through_both_predictions() -> None:
    type_logits, prediction, target_types, target_coords = _inputs()

    OutlineLoss()(type_logits, prediction, target_types, target_coords).backward()

    assert type_logits.grad is not None
    assert prediction.grad is not None
    assert torch.count_nonzero(type_logits.grad[-1]) == 0
    assert torch.count_nonzero(prediction.grad[-1]) == 0


def test_outline_loss_is_zero_for_all_padding() -> None:
    type_logits = torch.zeros((2, len(ElementType)), requires_grad=True)
    prediction = torch.zeros((2, 6), requires_grad=True)
    target_types = torch.tensor([ElementType.PAD, ElementType.PAD])
    target_coords = torch.zeros_like(prediction)

    loss = OutlineLoss()(type_logits, prediction, target_types, target_coords)
    loss.backward()

    assert loss.item() == 0.0
    assert type_logits.grad is not None
    assert prediction.grad is not None
    assert torch.count_nonzero(type_logits.grad) == 0
    assert torch.count_nonzero(prediction.grad) == 0


@pytest.mark.parametrize(
    ("logits_shape", "match"),
    [
        ((2, 3, len(ElementType)), "without its last dimension"),
        ((3, len(ElementType) - 1), "must have shape"),
    ],
)
def test_outline_loss_rejects_misaligned_type_logits(
    logits_shape: tuple[int, ...],
    match: str,
) -> None:
    _, prediction, target_types, target_coords = _inputs()

    with pytest.raises(ValueError, match=match):
        font_functional.outline_loss(
            torch.zeros(logits_shape), prediction, target_types, target_coords
        )


def test_outline_loss_rejects_invalid_reduction() -> None:
    with pytest.raises(ValueError, match="not a valid value"):
        font_functional.outline_loss(
            *_inputs(),
            reduction="invalid",  # ty: ignore[invalid-argument-type]
        )
