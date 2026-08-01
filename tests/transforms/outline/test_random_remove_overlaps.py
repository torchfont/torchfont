import numpy as np
import pytest
import torch

from torchfont import _torchfont
from torchfont.io import ElementType
from torchfont.transforms import Outline, RandomRemoveOverlaps


def _rectangles(
    rectangles: list[tuple[float, float]], *, closed: list[bool] | None = None
) -> tuple[torch.Tensor, torch.Tensor]:
    types: list[int] = []
    coords: list[list[float]] = []
    closed = closed or [True] * len(rectangles)
    for (x0, x1), is_closed in zip(rectangles, closed, strict=True):
        types.extend(
            [
                ElementType.MOVE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.CLOSE.value if is_closed else ElementType.LINE_TO.value,
            ]
        )
        coords.extend(
            [
                [0, 0, 0, 0, x0, 0],
                [0, 0, 0, 0, x1, 0],
                [0, 0, 0, 0, x1, 2],
                [0, 0, 0, 0, x0, 2],
                [0, 0, 0, 0, 0 if is_closed else x0, 0],
            ]
        )
    types.append(ElementType.END.value)
    coords.append([0, 0, 0, 0, 0, 0])
    return torch.tensor(types), torch.tensor(coords, dtype=torch.float32)


def _four_squares() -> tuple[torch.Tensor, torch.Tensor]:
    return _rectangles([(0.0, 2.0), (1.0, 3.0), (10.0, 12.0), (11.0, 13.0)])


def test_random_remove_overlaps_can_merge_multiple_groups() -> None:
    types, coords = _four_squares()
    torch.manual_seed(3)
    output = RandomRemoveOverlaps()(Outline(types, coords))
    out_types, out_coords = output.types, output.coords

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 2
    assert out_types.tolist().count(ElementType.CLOSE.value) == 2
    assert out_coords[:, 4].min() == 0
    assert out_coords[:, 4].max() == 13


def test_random_remove_overlaps_can_select_only_one_group() -> None:
    types, coords = _four_squares()
    torch.manual_seed(0)
    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 3


def test_random_remove_overlaps_selects_at_least_one_group() -> None:
    types, coords = _four_squares()
    # The first two values for this seed are both above the selection threshold.
    torch.manual_seed(4)
    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 3


def test_random_remove_overlaps_uses_connected_components() -> None:
    # The first and third rectangles do not intersect, but both intersect the second.
    types, coords = _rectangles([(0.0, 2.0), (1.0, 3.0), (2.5, 4.5)])

    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 1


def test_random_remove_overlaps_excludes_open_subpaths() -> None:
    types, coords = _rectangles(
        [(0.0, 2.0), (1.0, 3.0), (1.25, 1.75)], closed=[True, True, False]
    )

    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 2
    assert out_types.tolist().count(ElementType.CLOSE.value) == 1


def test_random_remove_overlaps_is_reproducible() -> None:
    outline = Outline(*_four_squares())
    torch.manual_seed(0)
    output1 = RandomRemoveOverlaps()(outline)
    torch.manual_seed(0)
    output2 = RandomRemoveOverlaps()(outline)

    assert torch.equal(output1.types, output2.types)
    assert torch.equal(output1.coords, output2.coords)


def test_random_remove_overlaps_leaves_non_candidates_unchanged() -> None:
    types, coords = _four_squares()
    separated_types = torch.cat([types[:5], types[10:15], types[-1:]])
    separated_coords = torch.cat([coords[:5], coords[10:15], coords[-1:]])

    output = RandomRemoveOverlaps()(Outline(separated_types, separated_coords))

    assert torch.equal(output.types, separated_types)
    assert torch.equal(output.coords, separated_coords)


def test_random_remove_overlaps_native_rejects_too_few_random_values() -> None:
    types, coords = _four_squares()

    with pytest.raises(
        ValueError, match="random_values length must be at least types length"
    ):
        _torchfont.random_remove_overlaps(
            types.numpy(), coords.reshape(-1).numpy(), np.zeros(1, dtype=np.float32)
        )
