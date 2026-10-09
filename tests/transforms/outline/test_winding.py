from typing import Literal

import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import NormalizeWinding, RandomWinding
from torchfont.transforms import functional as F  # noqa: N812

_SQUARE = [(0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (0.0, 1.0)]
_HOLE = [(0.2, 0.2), (0.2, 0.8), (0.8, 0.8), (0.8, 0.2)]
_ACCENT = [(2.0, 0.0), (3.0, 0.0), (3.0, 1.0), (2.0, 1.0)]


@pytest.mark.parametrize(
    ("types", "coords"),
    [
        pytest.param(
            [
                ElementType.MOVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CLOSE,
                ElementType.MOVE_TO,
                ElementType.LINE_TO,
                ElementType.LINE_TO,
                ElementType.CLOSE,
                ElementType.END,
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.49184057116508484, 0.17413905262947083],
                [
                    0.2575080394744873,
                    0.14203429222106934,
                    0.3305726647377014,
                    0.19542954862117767,
                    0.4807370901107788,
                    0.21822018921375275,
                ],
                [
                    0.823022723197937,
                    0.1344170868396759,
                    0.8491471409797668,
                    0.01251194253563881,
                    0.8996511697769165,
                    -0.07163329422473907,
                ],
                [
                    0.8721655607223511,
                    -0.10895946621894836,
                    0.8591591715812683,
                    -0.10571609437465668,
                    0.851406455039978,
                    -0.09315772354602814,
                ],
                [
                    0.8388388752937317,
                    -0.07111388444900513,
                    0.8277572989463806,
                    -0.04681894928216934,
                    0.8181959390640259,
                    -0.020317912101745605,
                ],
                [
                    0.7903509140014648,
                    0.059744928032159805,
                    0.7778959274291992,
                    0.15937396883964539,
                    0.7976130247116089,
                    0.26806843280792236,
                ],
                [
                    1.2883715629577637,
                    0.6053322553634644,
                    1.333451509475708,
                    0.5418708920478821,
                    1.3216395378112793,
                    0.4652225077152252,
                ],
                [
                    1.3043588399887085,
                    0.3460480272769928,
                    1.178917407989502,
                    0.28727298974990845,
                    0.966917872428894,
                    0.2554216980934143,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.7189663648605347, -0.23250626027584076],
                [0.0, 0.0, 0.0, 0.0, 0.848440408706665, 0.560584306716919],
                [0.0, 0.0, 0.0, 0.0, 0.7240208387374878, -0.23315037786960602],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            id="narrow_curve_overlap",
        ),
        pytest.param(
            [
                ElementType.MOVE_TO,
                ElementType.LINE_TO,
                ElementType.LINE_TO,
                ElementType.CURVE_TO,
                ElementType.CLOSE,
                ElementType.MOVE_TO,
                ElementType.LINE_TO,
                ElementType.CLOSE,
                ElementType.END,
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.6454120874404907, 0.1450488567352295],
                [0.0, 0.0, 0.0, 0.0, 0.7224063277244568, 0.0622008815407753],
                [0.0, 0.0, 0.0, 0.0, 0.6501503586769104, 0.1360609084367752],
                [
                    0.6519293189048767,
                    0.1448376327753067,
                    0.6397809982299805,
                    0.13942064344882965,
                    0.6561112999916077,
                    0.12876716256141663,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.6222383379936218, -0.020540665835142136],
                [0.0, 0.0, 0.0, 0.0, 0.7241284251213074, 0.07717401534318924],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            id="empty_retraced_line",
        ),
        pytest.param(
            [
                ElementType.MOVE_TO,
                ElementType.LINE_TO,
                ElementType.CURVE_TO,
                ElementType.LINE_TO,
                ElementType.CLOSE,
                ElementType.MOVE_TO,
                ElementType.LINE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CLOSE,
                ElementType.END,
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 1.2384127378463745, 0.2109431028366089],
                [0.0, 0.0, 0.0, 0.0, 0.400463342666626, 1.1590471267700195],
                [
                    0.4140649139881134,
                    1.1999783515930176,
                    0.42328497767448425,
                    1.199746012687683,
                    0.43465107679367065,
                    1.1868858337402344,
                ],
                [0.0, 0.0, 0.0, 0.0, 1.2726002931594849, 0.23878183960914612],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.527930498123169, 0.5944125056266785],
                [0.0, 0.0, 0.0, 0.0, 1.1113990545272827, 0.3546536862850189],
                [
                    0.9016837477684021,
                    0.5180995464324951,
                    0.9287294149398804,
                    0.6305182576179504,
                    0.9163005352020264,
                    0.7829436659812927,
                ],
                [
                    0.9567244648933411,
                    0.7926396727561951,
                    0.9646806716918945,
                    0.7836377024650574,
                    0.966480016708374,
                    0.7674106955528259,
                ],
                [
                    0.9825778007507324,
                    0.5505218505859375,
                    0.9223877787590027,
                    0.4030971825122833,
                    0.8056935667991638,
                    0.3080742061138153,
                ],
                [
                    0.6976604461669922,
                    0.2201038897037506,
                    0.5846205949783325,
                    0.21762284636497498,
                    0.49937641620635986,
                    0.31407296657562256,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            id="coincident_curves",
        ),
        pytest.param(
            [
                ElementType.MOVE_TO,
                ElementType.CURVE_TO,
                ElementType.CLOSE,
                ElementType.MOVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CURVE_TO,
                ElementType.CLOSE,
                ElementType.END,
            ],
            [
                [0.0, 0.0, 0.0, 0.0, 0.7117435932159424, 0.44694650173187256],
                [
                    0.8042411804199219,
                    0.6187016367912292,
                    0.7845208644866943,
                    0.625393271446228,
                    0.7716514468193054,
                    0.6061556339263916,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.5993096232414246, 0.014308704994618893],
                [
                    0.9301462769508362,
                    0.6402876973152161,
                    0.880584716796875,
                    0.7423208951950073,
                    0.7063956260681152,
                    0.7253797650337219,
                ],
                [
                    0.5506381988525391,
                    0.6863166093826294,
                    0.5346099138259888,
                    0.6728596687316895,
                    0.5351444482803345,
                    0.6555260419845581,
                ],
                [
                    0.5361248254776001,
                    0.6477278470993042,
                    0.5397790670394897,
                    0.6606171131134033,
                    0.5355129241943359,
                    0.6543527245521545,
                ],
                [
                    0.8467338681221008,
                    0.6660656929016113,
                    0.8693321943283081,
                    0.5877982974052429,
                    0.8817723393440247,
                    0.4240953028202057,
                ],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            id="disjoint_curves",
        ),
    ],
)
@pytest.mark.parametrize("fill_rule", ["winding", "even_odd"])
def test_difficult_curve_intersections_preserve_fill(
    types: list[ElementType],
    coords: list[list[float]],
    fill_rule: Literal["winding", "even_odd"],
) -> None:
    outline = Outline(
        torch.tensor(types, dtype=torch.long),
        torch.tensor(coords, dtype=torch.float32),
    )
    reversed_outline = F.reverse_winding_groups(outline, torch.tensor([True, True]))
    restored = F.reverse_winding_groups(reversed_outline, torch.tensor([True, True]))
    assert torch.equal(restored.types, outline.types)
    assert torch.equal(restored.coords, outline.coords)
    torch.manual_seed(0)
    results = [
        F.normalize_winding(outline, clockwise=clockwise) for clockwise in (True, False)
    ]
    results.append(RandomWinding()(outline))
    results.extend(
        F.reverse_winding_groups(outline, torch.tensor(mask))
        for mask in ([False, False], [True, False], [False, True], [True, True])
    )
    original = F.render_bitmap(outline, 128, mode="bbox", fill_rule=fill_rule)
    for result in results:
        rendered = F.render_bitmap(result, 128, mode="bbox", fill_rule=fill_rule)
        assert not (
            ((original == 255) & (rendered == 0))
            | ((original == 0) & (rendered == 255))
        ).any()


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


def test_disjoint_contours_with_overlapping_bounds_share_a_group() -> None:
    outline = _outline([(0, 0), (2, 0), (0, 2)], [(2, 2), (2, 0.8), (0.8, 2)])
    assert _signs(F.normalize_winding(outline)) == [-1, 1]
    assert _signs(F.reverse_winding_groups(outline, torch.tensor([True, False]))) == [
        -1,
        1,
    ]


def test_contour_in_a_concave_gap_shares_a_bounding_box_group() -> None:
    outline = _outline(
        [(0, 0), (3, 0), (3, 1), (1, 1), (1, 2), (3, 2), (3, 3), (0, 3)],
        [(2, 1.2), (2.5, 1.2), (2.5, 1.8), (2, 1.8)],
    )
    output = F.reverse_winding_groups(outline, torch.tensor([True, False]))
    assert _signs(output) == [-1, -1]


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


def test_shared_diagonal_with_overlapping_bounds_forms_one_group() -> None:
    outline = _outline([(0, 0), (1, 0), (0, 1)], [(1, 1), (0, 1), (1, 0)])
    output = F.reverse_winding_groups(outline, torch.tensor([True, False]))
    assert _signs(output) == [-1, -1]


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
def test_zero_signed_area_contours_are_grouped_by_bounds(*, filled: bool) -> None:
    points = (
        [(0.2, 0.2), (0.8, 0.8), (0.2, 0.8), (0.8, 0.2)]
        if filled
        else [(0.2, 0.2), (0.8, 0.2), (0.2, 0.8), (0.8, 0.2)]
    )
    outline = _outline(_SQUARE, points)
    mask = torch.tensor([True])
    output = F.reverse_winding_groups(outline, mask)
    for fill_rule in ("winding", "even_odd"):
        assert torch.equal(
            F.render_bitmap(output, 96, fill_rule=fill_rule, antialias=False),
            F.render_bitmap(outline, 96, fill_rule=fill_rule, antialias=False),
        )


def test_retraced_cubic_with_roundoff_preserves_fill_in_a_bounding_box_group() -> None:
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
    output = F.reverse_winding_groups(outline, torch.tensor([True]))
    assert _signs(output)[0] == -1
    for fill_rule in ("winding", "even_odd"):
        assert torch.equal(
            F.render_bitmap(output, 128, fill_rule=fill_rule, antialias=False),
            F.render_bitmap(outline, 128, fill_rule=fill_rule, antialias=False),
        )


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
    first, second = RandomWinding()([outline, outline])
    assert _signs(first) == [1, -1, -1]
    assert torch.equal(first.coords, second.coords)
    torch.manual_seed(0)
    repeated = RandomWinding()(outline)
    assert torch.equal(first.coords, repeated.coords)


@pytest.mark.parametrize("reverse_body", [False, True])
@pytest.mark.parametrize("reverse_accent", [False, True])
def test_random_directions_do_not_depend_on_input_winding(
    *, reverse_body: bool, reverse_accent: bool
) -> None:
    square = list(reversed(_SQUARE)) if reverse_body else _SQUARE
    hole = list(reversed(_HOLE)) if reverse_body else _HOLE
    accent = list(reversed(_ACCENT)) if reverse_accent else _ACCENT
    outline = _outline(square, hole, accent)
    torch.manual_seed(0)
    output = RandomWinding()(outline)
    assert _signs(output) == [1, -1, -1]


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
    reversed_outline = F.reverse_winding_groups(outline, torch.tensor([True]))
    assert not torch.equal(reversed_outline.coords, outline.coords)
    for output in (
        F.normalize_winding(outline, clockwise=False),
        RandomWinding()(outline),
        reversed_outline,
    ):
        assert torch.equal(F.render_bitmap(output, 96), original)
