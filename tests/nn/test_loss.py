import pytest
import torch
from torch.nn import functional as torch_functional

from torchfont import TYPE_DIM, ElementType
from torchfont.nn import OutlineLoss
from torchfont.nn import functional as font_functional


def _inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    type_logits = torch.zeros((3, TYPE_DIM), requires_grad=True)
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
    expected_coord_loss = font_functional.coordinate_loss(
        prediction, target_types, target_coords
    )
    assert torch.allclose(loss, 2 * expected_type_loss + 3 * expected_coord_loss)


def test_outline_loss_module_delegates_to_functional_loss() -> None:
    inputs = _inputs()
    criterion = OutlineLoss(type_weight=0.25, coordinate_weight=2.0)

    actual = criterion(*inputs)
    expected = font_functional.outline_loss(
        *inputs,
        type_weight=0.25,
        coordinate_weight=2.0,
    )

    assert torch.equal(actual, expected)
    assert repr(criterion) == "OutlineLoss(type_weight=0.25, coordinate_weight=2.0)"


def test_outline_loss_backpropagates_through_both_predictions() -> None:
    type_logits, prediction, target_types, target_coords = _inputs()

    OutlineLoss()(type_logits, prediction, target_types, target_coords).backward()

    assert type_logits.grad is not None
    assert prediction.grad is not None
    assert torch.count_nonzero(type_logits.grad[-1]) == 0
    assert torch.count_nonzero(prediction.grad[-1]) == 0


def test_outline_loss_is_zero_for_all_padding() -> None:
    type_logits = torch.zeros((2, TYPE_DIM), requires_grad=True)
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
        ((2, 3, TYPE_DIM), "without its last dimension"),
        ((3, TYPE_DIM - 1), "must have shape"),
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
