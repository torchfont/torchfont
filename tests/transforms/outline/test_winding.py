from typing import Literal

import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import NormalizeWinding, RandomReverseWinding
from torchfont.transforms import functional as F  # noqa: N812

_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
_HOLE = [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)]
_ACCENT = [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]


def _outline(*contours: list[tuple[float, float]]) -> Outline:
    types = []
    coords = []
    for points in contours:
        types.extend(
            [ElementType.MOVE_TO]
            + [ElementType.LINE_TO] * (len(points) - 1)
            + [ElementType.CLOSE]
        )
        coords.extend([[0, 0, 0, 0, x, y] for x, y in points] + [[0] * 6])
    return Outline(
        torch.tensor([*types, ElementType.END]),
        torch.tensor([*coords, [0] * 6], dtype=torch.float32),
    )


def _signs(outline: Outline) -> list[int]:
    result = []
    for contour in F.split_subpaths(outline):
        points = contour.coords[contour.types <= ElementType.LINE_TO, 4:]
        following = points.roll(-1, 0)
        area = (points[:, 0] * following[:, 1] - points[:, 1] * following[:, 0]).sum()
        result.append(-1 if area < 0 else 1)
    return result


@pytest.mark.parametrize("clockwise", [True, False])
def test_normalizes_disjoint_groups_with_their_holes(*, clockwise: bool) -> None:
    outline = _outline(_SQUARE, _HOLE, list(reversed(_ACCENT)))
    output = NormalizeWinding(clockwise=clockwise)(outline)
    sign = -1 if clockwise else 1
    assert _signs(output) == [sign, -sign, sign]
    repeated = F.normalize_winding(output, clockwise=clockwise)
    assert torch.equal(repeated.coords, output.coords)
    assert torch.equal(repeated.types, output.types)


@pytest.mark.parametrize("clockwise", [True, False])
def test_normalizes_uniformly_oriented_contours(*, clockwise: bool) -> None:
    output = F.normalize_winding(_outline(_SQUARE, _ACCENT), clockwise=clockwise)
    sign = -1 if clockwise else 1
    assert _signs(output) == [sign, sign]


def test_largest_absolute_area_determines_group_direction() -> None:
    outline = _outline(list(reversed(_HOLE)), list(reversed(_SQUARE)))
    assert _signs(F.normalize_winding(outline)) == [1, -1]


def test_equal_absolute_areas_prefer_the_first_contour() -> None:
    outline = _outline(_SQUARE, list(reversed(_SQUARE)))
    assert _signs(F.normalize_winding(outline)) == [-1, 1]


def test_disjoint_contours_with_overlapping_bounds_are_independent() -> None:
    outline = _outline([(0, 0), (2, 0), (0, 2)], [(2, 2), (2, 0.8), (0.8, 2)])
    assert _signs(F.normalize_winding(outline)) == [-1, -1]
    assert _signs(F.reverse_winding_groups(outline, torch.tensor([True, False]))) == [
        -1,
        -1,
    ]


def test_contour_in_a_concave_gap_is_independent() -> None:
    outline = _outline(
        [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)],
        [(2, 1.2), (2.5, 1.2), (2.5, 1.8), (2, 1.8)],
    )
    output = F.reverse_winding_groups(outline, torch.tensor([True, False]))
    assert _signs(output) == [-1, 1]


@pytest.mark.parametrize(
    ("mask", "expected"),
    [([True, False], [-1, -1, 1]), ([False, True], [1, 1, -1])],
)
def test_group_order_follows_input_when_spatial_order_is_reversed(
    mask: list[bool], expected: list[int]
) -> None:
    outline = _outline(_ACCENT, _HOLE, _SQUARE)
    output = F.reverse_winding_groups(outline, torch.tensor(mask))
    assert _signs(output) == expected


def test_shared_diagonal_is_not_a_filled_intersection() -> None:
    outline = _outline([(0, 0), (1, 0), (0, 1)], [(1, 1), (0, 1), (1, 0)])
    output = F.reverse_winding_groups(outline, torch.tensor([True, False]))
    assert _signs(output) == [-1, 1]


def test_crossing_contours_without_contained_vertices_form_one_group() -> None:
    outline = _outline(
        [(-1, 0.4), (2, 0.4), (2, 0.6), (-1, 0.6)],
        [(0.4, -1), (0.6, -1), (0.6, 2), (0.4, 2)],
    )
    output = F.reverse_winding_groups(outline, torch.tensor([True]))
    assert _signs(output) == [-1, -1]


def test_transitive_intersections_form_one_group() -> None:
    right = [(x + 1.2, y) for x, y in reversed(_SQUARE)]
    bridge = [(x + 0.5, y) for x, y in _SQUARE]
    outline = _outline(right, _SQUARE, bridge)
    output = F.reverse_winding_groups(outline, torch.tensor([True]))
    assert _signs(output) == [1, -1, -1]


@pytest.mark.parametrize("filled", [True, False])
def test_zero_signed_area_contours_are_grouped_by_fill(*, filled: bool) -> None:
    points = (
        [(0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)]
        if filled
        else [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.2)]
    )
    outline = _outline(_SQUARE, points)
    if not filled:
        with pytest.raises(ValueError, match="one value per winding group"):
            F.reverse_winding_groups(outline, torch.tensor([True]))
    mask = torch.tensor([True] if filled else [True, False])
    output = F.reverse_winding_groups(outline, mask)
    for fill_rule in ("winding", "even_odd"):
        assert torch.equal(
            F.render_bitmap(output, 96, fill_rule=fill_rule, antialias=False),
            F.render_bitmap(outline, 96, fill_rule=fill_rule, antialias=False),
        )


def test_retraced_cubic_with_roundoff_has_no_filled_intersection() -> None:
    points = torch.tensor(
        [
            [0.41692355275154114, 0.11778823286294937],
            [0.4754806160926819, 0.5988467335700989],
            [0.8568907380104065, 0.4482608735561371],
            [0.48851311206817627, 0.5152921080589294],
        ],
        dtype=torch.float32,
    )
    coords = torch.zeros((5, 6))
    coords[0, 4:] = points[0]
    coords[1] = points[1:].flatten()
    coords[2] = points[[2, 1, 0]].flatten()
    square = _outline(_SQUARE)
    outline = Outline(
        torch.cat(
            (
                square.types[:-1],
                torch.tensor(
                    [
                        ElementType.MOVE_TO,
                        ElementType.CURVE_TO,
                        ElementType.CURVE_TO,
                        ElementType.CLOSE,
                        ElementType.END,
                    ]
                ),
            )
        ),
        torch.cat((square.coords[:-1], coords)),
    )
    with pytest.raises(ValueError, match="one value per winding group"):
        F.reverse_winding_groups(outline, torch.tensor([True]))
    output = F.reverse_winding_groups(outline, torch.tensor([True, False]))
    assert _signs(output)[0] == -1
    assert torch.equal(output.coords[5:], outline.coords[5:])


@pytest.mark.parametrize("fill_rule", ["winding", "even_odd"])
@pytest.mark.parametrize(
    "case", ["hole", "touching_hole", "redundant", "overlap_chain"]
)
def test_preserves_fill(fill_rule: Literal["winding", "even_odd"], case: str) -> None:
    contours = {
        "hole": [_SQUARE, _HOLE, _ACCENT],
        "touching_hole": [_SQUARE, [(0, 0.2), (0, 0.8), (0.8, 0.8), (0.8, 0.2)]],
        "redundant": [_SQUARE, list(reversed(_HOLE))],
        "overlap_chain": [
            _SQUARE,
            [(x + 0.5, y) for x, y in _SQUARE],
            [(x + 1.2, y) for x, y in reversed(_SQUARE)],
        ],
    }[case]
    outline = _outline(*contours)
    normalized = F.normalize_winding(outline)
    reversed_groups = F.reverse_winding_groups(outline, torch.tensor([True, False]))
    original = F.render_bitmap(outline, 96, fill_rule=fill_rule, antialias=False)
    for result in (normalized, reversed_groups):
        assert torch.equal(
            original, F.render_bitmap(result, 96, fill_rule=fill_rule, antialias=False)
        )


def test_random_groups_are_independent_and_shared_between_outlines() -> None:
    outline = _outline(_SQUARE, _HOLE, _ACCENT)
    torch.manual_seed(0)
    first, second = RandomReverseWinding()([outline, outline])
    assert _signs(first) == [-1, 1, 1]
    assert torch.equal(first.coords, second.coords)
    torch.manual_seed(0)
    repeated = RandomReverseWinding()(outline)
    assert torch.equal(first.coords, repeated.coords)


@pytest.mark.parametrize("p", [0.0, 1.0])
def test_probability_boundaries(p: float) -> None:
    output = RandomReverseWinding(p)(_outline(_SQUARE, _HOLE, _ACCENT))
    assert _signs(output) == ([1, -1, 1] if p == 0 else [-1, 1, -1])


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_invalid_probability(p: float) -> None:
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        RandomReverseWinding(p)


@pytest.mark.parametrize("selected", [False, True])
def test_mask_must_cover_every_group(*, selected: bool) -> None:
    with pytest.raises(ValueError, match="one value per winding group"):
        F.reverse_winding_groups(_outline(_SQUARE, _ACCENT), torch.tensor([selected]))


@pytest.mark.parametrize("selected", [False, True])
def test_mask_covers_groups_and_ignores_extra_entries(*, selected: bool) -> None:
    outline = _outline(_SQUARE, _HOLE, _ACCENT)
    short = F.reverse_winding_groups(outline, torch.tensor([selected, selected]))
    long = F.reverse_winding_groups(
        outline, torch.tensor([selected, selected, selected, not selected])
    )
    assert torch.equal(long.types, short.types)
    assert torch.equal(long.coords, short.coords)


def test_normalization_keeps_an_independent_zero_area_contour() -> None:
    outline = _outline(_SQUARE, [(2, 0), (3, 0)])
    output = F.normalize_winding(outline)
    assert _signs(output)[0] == -1
    assert torch.equal(output.coords[5:], outline.coords[5:])


def test_normalization_reverses_closed_contours_but_keeps_open_contours() -> None:
    outline = _outline(list(reversed(_SQUARE)), list(reversed(_ACCENT)))
    close = (outline.types == ElementType.CLOSE).nonzero().flatten()[-1]
    keep = torch.arange(outline.types.numel()) != close
    outline = Outline(outline.types[keep], outline.coords[keep])
    assert _signs(F.normalize_winding(outline, clockwise=False)) == [1, -1]


def test_uniform_reversal_keeps_a_group_with_an_open_contour() -> None:
    outline = _outline(_SQUARE, _HOLE)
    close = (outline.types == ElementType.CLOSE).nonzero().flatten()[-1]
    keep = torch.arange(outline.types.numel()) != close
    outline = Outline(outline.types[keep], outline.coords[keep])
    output = F.reverse_winding_groups(outline, torch.tensor([True, True]))
    assert torch.equal(output.types, outline.types)
    assert torch.equal(output.coords, outline.coords)


@pytest.mark.parametrize("zero_area", [False, True], ids=["empty", "zero_area"])
def test_empty_and_zero_area_outlines_are_unchanged(*, zero_area: bool) -> None:
    outline = _outline([(0, 0), (1, 0)]) if zero_area else _outline()
    mask = torch.zeros(int(zero_area), dtype=torch.bool)
    for output in (
        F.normalize_winding(outline),
        F.reverse_winding_groups(outline, mask),
    ):
        assert torch.equal(output.types, outline.types)
        assert torch.equal(output.coords, outline.coords)


def test_open_contour_keeps_only_its_interacting_group_unchanged() -> None:
    outline = _outline(_SQUARE, _HOLE, _ACCENT)
    # Open the inner contour: keep the interacting group and normalize the accent.
    close_indices = (outline.types == ElementType.CLOSE).nonzero().flatten()
    keep = torch.arange(outline.types.numel()) != close_indices[1]
    outline = Outline(outline.types[keep], outline.coords[keep])
    assert _signs(F.normalize_winding(outline)) == [1, -1, -1]
    assert _signs(F.reverse_winding_groups(outline, torch.tensor([True, True]))) == [
        1,
        -1,
        -1,
    ]


@pytest.mark.parametrize("fixture", ["quad_outline", "cubic_outline"])
def test_preserves_curve_geometry(request: pytest.FixtureRequest, fixture: str) -> None:
    outline = Outline(*request.getfixturevalue(fixture))
    crossing = _outline([(0.4, 0.1), (0.6, 0.1), (0.6, 0.7), (0.4, 0.7)])
    outline = Outline(
        torch.cat((outline.types[:-1], crossing.types)),
        torch.cat((outline.coords[:-1], crossing.coords)),
    )
    original = F.render_bitmap(outline, 96)
    for output in (
        F.normalize_winding(outline, clockwise=False),
        RandomReverseWinding(1.0)(outline),
        F.reverse_winding_groups(outline, torch.tensor([True])),
    ):
        assert not torch.equal(output.coords, outline.coords)
        assert torch.equal(F.render_bitmap(output, 96), original)
