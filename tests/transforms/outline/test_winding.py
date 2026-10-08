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


def test_disjoint_contours_with_overlapping_bounds_are_independent() -> None:
    outline = _outline([(0, 0), (2, 0), (0, 2)], [(2, 2), (2, 0.8), (0.8, 2)])
    assert _signs(F.normalize_winding(outline)) == [-1, -1]
    assert _signs(F.reverse_winding_groups(outline, torch.tensor([True, False]))) == [
        -1,
        -1,
    ]


@pytest.mark.parametrize("fill_rule", ["winding", "even_odd"])
@pytest.mark.parametrize("case", ["hole", "redundant", "overlap_chain"])
def test_preserves_fill(fill_rule: Literal["winding", "even_odd"], case: str) -> None:
    contours = {
        "hole": [_SQUARE, _HOLE, _ACCENT],
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


def test_mask_must_cover_every_group() -> None:
    with pytest.raises(ValueError, match="one value per winding group"):
        F.reverse_winding_groups(_outline(_SQUARE, _ACCENT), torch.tensor([True]))


def test_empty_zero_area_and_open_groups() -> None:
    for outline in (_outline(), _outline([(0, 0), (1, 0)])):
        output = F.normalize_winding(outline)
        assert torch.equal(output.types, outline.types)
        assert torch.equal(output.coords, outline.coords)
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
    original = F.render_bitmap(outline, 96)
    for output in (F.normalize_winding(outline), RandomReverseWinding(1.0)(outline)):
        assert torch.equal(F.render_bitmap(output, 96), original)
