import numpy as np
import pytest
import torch

from torchfont import ElementType, Outline, _torchfont
from torchfont.transforms import RandomRemoveOverlaps, functional


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


def _polygons(*polygons: list[tuple[float, float]]) -> Outline:
    types: list[int] = []
    coords: list[list[float]] = []
    for polygon in polygons:
        types.extend([ElementType.MOVE_TO.value])
        types.extend([ElementType.LINE_TO.value] * (len(polygon) - 1))
        types.append(ElementType.CLOSE.value)
        coords.extend([[0, 0, 0, 0, x, y] for x, y in polygon])
        coords.append([0, 0, 0, 0, 0, 0])
    types.append(ElementType.END.value)
    coords.append([0, 0, 0, 0, 0, 0])
    return Outline(torch.tensor(types), torch.tensor(coords, dtype=torch.float32))


def test_random_remove_overlaps_ignores_overlapping_bounds() -> None:
    outline = _polygons([(0, 0), (2, 0), (0, 2)], [(2, 2), (2, 0.8), (0.8, 2)])

    output = functional.remove_overlap_groups(
        outline, torch.zeros(outline.types.size(0))
    )

    assert torch.equal(output.types, outline.types)
    assert torch.equal(
        functional.render_bitmap(output), functional.render_bitmap(outline)
    )


@pytest.mark.parametrize("offset", [(2, 0), (2, 2)])
def test_random_remove_overlaps_ignores_boundary_contacts(
    offset: tuple[int, int],
) -> None:
    square: list[tuple[float, float]] = [(0, 0), (2, 0), (2, 2), (0, 2)]
    dx, dy = offset
    outline = _polygons(square, [(x + dx, y + dy) for x, y in square])

    output = RandomRemoveOverlaps()(outline)

    assert torch.equal(output.types, outline.types)
    assert torch.equal(output.coords, outline.coords)


def test_random_remove_overlaps_cancels_opposite_contours() -> None:
    square: list[tuple[float, float]] = [(0, 0), (2, 0), (2, 2), (0, 2)]
    outline = _polygons(square, list(reversed(square)))

    output = RandomRemoveOverlaps()(outline)

    assert output.types.tolist() == [ElementType.END.value]


def test_random_remove_overlaps_simplifies_single_self_intersection() -> None:
    outline = _polygons([(0, 0), (2, 2), (0, 2), (2, 0)])

    output = RandomRemoveOverlaps()(outline)
    expected = functional.remove_overlaps(outline)

    assert output.types.tolist().count(ElementType.MOVE_TO.value) == 2
    assert torch.equal(output.types, expected.types)
    assert torch.equal(output.coords, expected.coords)


@pytest.mark.parametrize("values", [[0.0, 1.0], [1.0, 0.0], [0.9, 0.8]])
def test_random_remove_overlaps_selects_single_contour_groups(
    values: list[float],
) -> None:
    bowtie = [(0.0, 0.0), (2.0, 2.0), (0.0, 2.0), (2.0, 0.0)]
    outline = _polygons(bowtie, [(x + 4, y) for x, y in bowtie])

    selection = torch.ones(outline.types.size(0))
    selection[:2] = torch.tensor(values)
    output = functional.remove_overlap_groups(outline, selection)

    assert output.types.tolist().count(ElementType.MOVE_TO.value) == 3
    start = 0 if values[0] > values[1] else len(output.types) - 6
    assert torch.equal(output.types[start : start + 5], outline.types[:5])
    source = outline.coords[:5] if values[0] > values[1] else outline.coords[5:10]
    assert torch.equal(output.coords[start : start + 5], source)


def test_random_remove_overlaps_simplifies_single_open_self_intersection() -> None:
    outline = _polygons([(0, 0), (2, 2), (0, 2), (2, 0)])
    outline = Outline(
        torch.cat([outline.types[:-2], outline.types[-1:]]),
        torch.cat([outline.coords[:-2], outline.coords[-1:]]),
    )

    output = RandomRemoveOverlaps()(outline)

    expected = functional.remove_overlaps(outline)
    assert output.types.tolist().count(ElementType.MOVE_TO.value) == 2
    assert torch.equal(output.types, expected.types)
    assert torch.equal(output.coords, expected.coords)


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
    torch.manual_seed(4)
    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 3


def test_random_remove_overlaps_uses_connected_components() -> None:
    types, coords = _rectangles([(0.0, 2.0), (1.0, 3.0), (2.5, 4.5)])

    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 1


def test_random_remove_overlaps_includes_open_subpaths() -> None:
    types, coords = _rectangles(
        [(0.0, 2.0), (1.0, 3.0), (1.25, 1.75)], closed=[True, True, False]
    )

    out_types = RandomRemoveOverlaps()(Outline(types, coords)).types

    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 1
    assert out_types.tolist().count(ElementType.CLOSE.value) == 1


def test_random_remove_overlaps_is_reproducible() -> None:
    outline = Outline(*_four_squares())
    torch.manual_seed(0)
    output1 = RandomRemoveOverlaps()(outline)
    torch.manual_seed(0)
    output2 = RandomRemoveOverlaps()(outline)

    assert torch.equal(output1.types, output2.types)
    assert torch.equal(output1.coords, output2.coords)


def test_random_remove_overlaps_verify_keeps_a_correct_result() -> None:
    outline = Outline(*_four_squares())
    torch.manual_seed(3)
    unverified = RandomRemoveOverlaps(verify=False)(outline)
    torch.manual_seed(3)
    verified = RandomRemoveOverlaps(verify=True)(outline)

    assert torch.equal(verified.types, unverified.types)
    assert torch.allclose(verified.coords, unverified.coords)


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
            types.numpy(),
            coords.reshape(-1).numpy(),
            np.zeros(1, dtype=np.float32),
            verify=False,
            verify_size=128,
        )
