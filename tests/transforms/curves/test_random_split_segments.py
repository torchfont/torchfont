import numpy as np
import pytest
import torch

from torchfont import _torchfont
from torchfont.io import ElementType
from torchfont.transforms import Outline, RandomSplitSegments


def _mixed_segments() -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.MOVE_TO.value,
            ElementType.QUAD_TO.value,
            ElementType.MOVE_TO.value,
            ElementType.CURVE_TO.value,
            ElementType.END.value,
        ]
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 2, 0],
            [0, 0, 0, 0, 0, 0],
            [1, 3, 2, 3, 3, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=torch.float32,
    )
    return types, coords


def test_random_split_segments_splits_each_segment_type_at_de_casteljau_midpoint() -> (
    None
):
    types, coords = _mixed_segments()
    selection_values = np.full(types.numel(), 0.1, dtype=np.float32)
    position_values = np.full(types.numel(), 0.5, dtype=np.float32)

    out_types, flat_coords = _torchfont.random_split_segments(
        types.numpy(),
        coords.reshape(-1).numpy(),
        selection_values,
        position_values,
        0.2,
        (0.2, 0.8),
    )
    out_coords = torch.from_numpy(flat_coords).view(-1, 6)

    assert out_types.tolist() == [
        ElementType.MOVE_TO.value,
        ElementType.LINE_TO.value,
        ElementType.LINE_TO.value,
        ElementType.MOVE_TO.value,
        ElementType.QUAD_TO.value,
        ElementType.QUAD_TO.value,
        ElementType.MOVE_TO.value,
        ElementType.CURVE_TO.value,
        ElementType.CURVE_TO.value,
        ElementType.END.value,
    ]
    assert torch.equal(out_coords[1], torch.tensor([0, 0, 0, 0, 1, 0.0]))
    assert torch.equal(out_coords[2], torch.tensor([0, 0, 0, 0, 2, 0.0]))
    assert torch.equal(out_coords[4], torch.tensor([0.5, 1, 0, 0, 1, 1.0]))
    assert torch.equal(out_coords[5], torch.tensor([1.5, 1, 0, 0, 2, 0.0]))
    assert torch.equal(out_coords[7], torch.tensor([0.5, 1.5, 1, 2.25, 1.5, 2.25]))
    assert torch.equal(out_coords[8], torch.tensor([2, 2.25, 2.5, 1.5, 3, 0.0]))


def test_random_split_segments_can_leave_every_segment_unchanged() -> None:
    types, coords = _mixed_segments()
    selection_values = np.full(types.numel(), 0.9, dtype=np.float32)
    position_values = np.full(types.numel(), 0.5, dtype=np.float32)

    out_types, _ = _torchfont.random_split_segments(
        types.numpy(),
        coords.reshape(-1).numpy(),
        selection_values,
        position_values,
        0.2,
        (0.2, 0.8),
    )

    assert out_types.tolist().count(ElementType.LINE_TO.value) == 1
    assert out_types.tolist().count(ElementType.QUAD_TO.value) == 1
    assert out_types.tolist().count(ElementType.CURVE_TO.value) == 1


def test_random_split_segments_probability_boundaries() -> None:
    types, coords = _mixed_segments()

    outline = Outline(types, coords)
    unchanged = RandomSplitSegments(split_probability=0.0)(outline)
    split = RandomSplitSegments(split_probability=1.0)(outline)

    assert torch.equal(unchanged.types, types)
    assert torch.equal(unchanged.coords, coords)
    assert split.types.numel() == types.numel() + 3


@pytest.mark.parametrize("split_probability", [-0.1, 1.1, float("nan")])
def test_random_split_segments_rejects_invalid_probability(
    split_probability: float,
) -> None:
    _types, _coords = _mixed_segments()

    with pytest.raises(ValueError, match="split_probability must be between 0 and 1"):
        RandomSplitSegments(split_probability=split_probability)


@pytest.mark.parametrize(
    "split_range",
    [(0.0, 0.8), (0.2, 1.0), (0.8, 0.2), (float("nan"), 0.8)],
)
def test_random_split_segments_rejects_invalid_range(
    split_range: tuple[float, float],
) -> None:
    _types, _coords = _mixed_segments()

    with pytest.raises(ValueError, match="split_range must satisfy"):
        RandomSplitSegments(split_range=split_range)


def test_random_split_segments_accepts_fixed_split_parameter() -> None:
    types, coords = _mixed_segments()

    output = RandomSplitSegments(
        split_probability=1.0,
        split_range=(0.5, 0.5),
    )(Outline(types, coords))

    assert output.types.numel() == types.numel() + 3
    assert output.coords[1, 4].item() == pytest.approx(1.0)


def test_random_split_segments_is_reproducible() -> None:
    outline = Outline(*_mixed_segments())
    torch.manual_seed(0)
    output1 = RandomSplitSegments()(outline)
    torch.manual_seed(0)
    output2 = RandomSplitSegments()(outline)

    assert torch.equal(output1.types, output2.types)
    assert torch.equal(output1.coords, output2.coords)


def test_random_split_segments_native_rejects_too_few_selection_values() -> None:
    types, coords = _mixed_segments()

    with pytest.raises(
        ValueError,
        match="selection_values and position_values lengths",
    ):
        _torchfont.random_split_segments(
            types.numpy(),
            coords.reshape(-1).numpy(),
            np.zeros(1, dtype=np.float32),
            np.zeros(types.numel(), dtype=np.float32),
            0.2,
            (0.2, 0.8),
        )
