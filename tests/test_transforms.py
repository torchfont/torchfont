import math
from collections.abc import Callable, Sequence

import pytest
import torch

from torchfont.datasets import GlyphSample
from torchfont.io import CommandType
from torchfont.transforms import (
    cubic_to_quad,
    merge_curves,
    patchify,
    quad_to_cubic,
    remove_overlaps,
    render_bitmap,
)

_ZERO_METRICS = torch.zeros(15, dtype=torch.float32)
_Point = tuple[float, float]
_CubicSeg = tuple[_Point, _Point, _Point, _Point]
_QuadSeg = tuple[_Point, _Point, _Point]


def _lerp(
    a: tuple[float, float], b: tuple[float, float], t: float
) -> tuple[float, float]:
    return (a[0] * (1.0 - t) + b[0] * t, a[1] * (1.0 - t) + b[1] * t)


def _line_path_to_tensors(points: list[_Point]) -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.tensor(
        [CommandType.MOVE_TO.value]
        + [CommandType.LINE_TO.value] * (len(points) - 1)
        + [CommandType.CLOSE.value, CommandType.END.value],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [[0.0, 0.0, 0.0, 0.0, x, y] for x, y in points] + [[0.0] * 6, [0.0] * 6],
        dtype=torch.float32,
    )
    return types, coords


def _quad_split(
    p0: _Point, p1: _Point, p2: _Point, t: float
) -> tuple[_QuadSeg, _QuadSeg]:
    q0, q1 = _lerp(p0, p1, t), _lerp(p1, p2, t)
    s = _lerp(q0, q1, t)
    return (p0, q0, s), (s, q1, p2)


def _split_quad(curve: _QuadSeg, ts: tuple[float, ...]) -> list[_QuadSeg]:
    pieces: list[_QuadSeg] = []
    current = curve
    previous_t = 0.0
    for t in ts:
        local_t = (t - previous_t) / (1.0 - previous_t)
        left, current = _quad_split(*current, local_t)
        pieces.append(left)
        previous_t = t
    pieces.append(current)
    return pieces


def _quad_segs_to_tensors(
    segs: Sequence[_QuadSeg],
) -> tuple[torch.Tensor, torch.Tensor]:
    types_list = (
        [CommandType.MOVE_TO.value]
        + [CommandType.QUAD_TO.value] * len(segs)
        + [CommandType.CLOSE.value, CommandType.END.value]
    )
    p0 = segs[0][0]
    coords_list: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, p0[0], p0[1]]]
    coords_list.extend([s[1][0], s[1][1], 0.0, 0.0, s[2][0], s[2][1]] for s in segs)
    coords_list += [[0.0] * 6, [0.0] * 6]
    return (
        torch.tensor(types_list, dtype=torch.long),
        torch.tensor(coords_list, dtype=torch.float32),
    )


def _casteljau_split(
    p0: _Point,
    p1: _Point,
    p2: _Point,
    p3: _Point,
    t: float,
) -> tuple[_CubicSeg, _CubicSeg]:
    q0, q1, q2 = _lerp(p0, p1, t), _lerp(p1, p2, t), _lerp(p2, p3, t)
    r0, r1 = _lerp(q0, q1, t), _lerp(q1, q2, t)
    s = _lerp(r0, r1, t)
    return (p0, q0, r0, s), (s, r1, q2, p3)


def _cubic_segs_to_tensors(
    segs: Sequence[_CubicSeg],
) -> tuple[torch.Tensor, torch.Tensor]:
    n = len(segs)
    types_list = (
        [CommandType.MOVE_TO.value]
        + [CommandType.CURVE_TO.value] * n
        + [CommandType.CLOSE.value, CommandType.END.value]
    )
    p0 = segs[0][0]
    coords_list: list[list[float]] = [[0.0, 0.0, 0.0, 0.0, p0[0], p0[1]]]
    coords_list.extend(
        [s[1][0], s[1][1], s[2][0], s[2][1], s[3][0], s[3][1]] for s in segs
    )
    coords_list += [[0.0] * 6, [0.0] * 6]
    return (
        torch.tensor(types_list, dtype=torch.long),
        torch.tensor(coords_list, dtype=torch.float32),
    )


_CUBIC_CURVES: list[_CubicSeg] = [
    ((0.0, 0.0), (1.0, 0.0), (1.0, 1.0), (2.0, 1.0)),
    ((0.0, 0.0), (0.5, 1.0), (1.5, 1.0), (2.0, 0.0)),
    ((0.0, 0.0), (0.0, 1.0), (1.0, 1.0), (1.0, 0.0)),
    ((0.0, 0.0), (2.0, 0.0), (0.0, 1.0), (2.0, 1.0)),
]


def _split_cubic(curve: _CubicSeg, ts: tuple[float, ...]) -> list[_CubicSeg]:
    pieces: list[_CubicSeg] = []
    current = curve
    previous_t = 0.0
    for t in ts:
        local_t = (t - previous_t) / (1.0 - previous_t)
        left, current = _casteljau_split(*current, local_t)
        pieces.append(left)
        previous_t = t
    pieces.append(current)
    return pieces


def _sub_cubic(curve: _CubicSeg, t1: float, t2: float) -> _CubicSeg:
    if t1 == 0.0:
        left, _ = _casteljau_split(*curve, t2)
        return left
    _, right = _casteljau_split(*curve, t1)
    t_adj = (t2 - t1) / (1.0 - t1)
    left, _ = _casteljau_split(*right, t_adj)
    return left


def _cubic_coords(curve: _CubicSeg) -> torch.Tensor:
    p1, p2, p3 = curve[1], curve[2], curve[3]
    return torch.tensor([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]], dtype=torch.float32)


def _assert_single_cubic_matches(
    types: torch.Tensor,
    coords: torch.Tensor,
    curve: _CubicSeg,
    *,
    atol: float = 1e-4,
) -> None:
    assert types.tolist().count(CommandType.CURVE_TO.value) == 1
    idx = types.tolist().index(CommandType.CURVE_TO.value)
    assert torch.allclose(coords[idx], _cubic_coords(curve), atol=atol)


def test_merge_curves_collinear_lines_are_merged() -> None:
    types, coords = _line_path_to_tensors(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    )

    out_types, out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(CommandType.LINE_TO.value) == 1
    line_idx = out_types.tolist().index(CommandType.LINE_TO.value)
    assert out_coords[line_idx, 4].item() == pytest.approx(3.0)
    assert out_coords[line_idx, 5].item() == pytest.approx(0.0)


@pytest.mark.parametrize(
    "points",
    [
        pytest.param([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], id="corner"),
        # Tiny angular bend, but far from the merged line in normalized coordinates.
        pytest.param(
            [(0.0, 0.0), (1_000.0, 0.0), (2_000.0, 0.5)], id="absolute-tolerance"
        ),
        pytest.param([(0.0, 0.0), (1.0, 0.0), (0.5, 0.0)], id="antiparallel"),
    ],
)
def test_merge_curves_non_mergeable_lines_stay_separate(points: list[_Point]) -> None:
    types, coords = _line_path_to_tensors(points)

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(CommandType.LINE_TO.value) == 2


@pytest.mark.parametrize(
    "ts",
    [(0.2,), (0.4,), (0.5,), (0.7,), (0.9,), (0.2, 0.6), (0.3, 0.7), (0.5, 0.8)],
)
@pytest.mark.parametrize("curve", _CUBIC_CURVES)
def test_merge_curves_split_cubics_roundtrip(
    ts: tuple[float, ...], curve: _CubicSeg
) -> None:
    types, coords = _cubic_segs_to_tensors(_split_cubic(curve, ts))

    out_types, out_coords = merge_curves(types, coords)

    _assert_single_cubic_matches(out_types, out_coords, curve)


@pytest.mark.parametrize("ts", [(0.2,), (0.5,), (0.8,), (0.2, 0.6), (0.3, 0.7)])
def test_merge_curves_split_quads_roundtrip(ts: tuple[float, ...]) -> None:
    curve: _QuadSeg = ((0.0, 0.0), (1.0, 2.0), (3.0, 0.0))
    types, coords = _quad_segs_to_tensors(_split_quad(curve, ts))

    out_types, out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(CommandType.QUAD_TO.value) == 1
    idx = out_types.tolist().index(CommandType.QUAD_TO.value)
    assert torch.allclose(
        out_coords[idx],
        torch.tensor([1.0, 2.0, 0.0, 0.0, 3.0, 0.0]),
        atol=1e-4,
    )


def test_merge_curves_incompatible_quads_are_not_merged() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.QUAD_TO.value,
            CommandType.QUAD_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(CommandType.QUAD_TO.value) == 2


def test_merge_curves_incompatible_cubics_are_not_merged() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.CURVE_TO.value,
            CommandType.CURVE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 1.0, 1.0, 1.0, 0.0],
            [2.0, 0.0, 2.0, 1.0, 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(CommandType.CURVE_TO.value) == 2


def test_merge_curves_preserves_structure_outside_contour() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.zeros(4, 6, dtype=torch.float32)
    coords[1, 4] = 1.0

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types[-1].item() == CommandType.END.value
    assert CommandType.CLOSE.value in out_types.tolist()


def test_merge_curves_multiple_contours() -> None:
    def _contour(ox: float) -> tuple[list[int], list[list[float]]]:
        t = [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
        ]
        c = [
            [0.0, 0.0, 0.0, 0.0, ox, 0.0],
            [0.0, 0.0, 0.0, 0.0, ox + 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, ox + 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        return t, c

    t1, c1 = _contour(0.0)
    t2, c2 = _contour(10.0)
    end_t = [CommandType.END.value]
    end_c = [[0.0] * 6]

    types = torch.tensor(t1 + t2 + end_t, dtype=torch.long)
    coords = torch.tensor(c1 + c2 + end_c, dtype=torch.float32)

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(CommandType.LINE_TO.value) == 2
    assert out_types.tolist().count(CommandType.MOVE_TO.value) == 2


@pytest.mark.parametrize("transform", [cubic_to_quad, merge_curves])
def test_variable_length_transforms_reject_mismatched_coords(
    transform: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
) -> None:
    types = torch.tensor(
        [CommandType.MOVE_TO.value, CommandType.CURVE_TO.value],
        dtype=torch.long,
    )
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="coords length"):
        transform(types, coords)


def test_quad_to_cubic_converts_quadratic_segments() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.QUAD_TO.value,
            CommandType.LINE_TO.value,
            CommandType.QUAD_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            [3.0, 2.0, 0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = quad_to_cubic(types, coords)

    expected_types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.CURVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CURVE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    expected_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            [8.0 / 3.0, 5.0 / 3.0, 10.0 / 3.0, 5.0 / 3.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(out_types, expected_types)
    assert torch.allclose(out_coords, expected_coords)


def test_quad_to_cubic_returns_inputs_when_no_quadratic_segments() -> None:
    types = torch.tensor(
        [CommandType.MOVE_TO.value, CommandType.LINE_TO.value, CommandType.END.value],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = quad_to_cubic(types, coords)

    assert out_types is types
    assert out_coords is coords
    assert out_types.device.type == "cpu"
    assert out_coords.device.type == "cpu"


def test_quad_to_cubic_supports_extra_leading_dimensions() -> None:
    types = torch.tensor(
        [
            [CommandType.MOVE_TO.value, CommandType.QUAD_TO.value],
            [CommandType.LINE_TO.value, CommandType.END.value],
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    sample = GlyphSample(
        types=types,
        coords=coords,
        style_idx=4,
        content_idx=5,
        metrics=_ZERO_METRICS,
        glyph_name="",
    )
    out_types, out_coords = quad_to_cubic(sample.types, sample.coords)

    assert out_types.shape == types.shape
    assert out_coords.shape == coords.shape
    assert out_types[0, 1].item() == CommandType.CURVE_TO.value
    assert torch.allclose(
        out_coords[0, 1],
        torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )


def test_quad_to_cubic_keeps_batched_sequences_independent() -> None:
    types = torch.tensor(
        [
            [
                CommandType.MOVE_TO.value,
                CommandType.LINE_TO.value,
                CommandType.END.value,
            ],
            [
                CommandType.QUAD_TO.value,
                CommandType.LINE_TO.value,
                CommandType.END.value,
            ],
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 7.0, 7.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
            ],
            [
                [0.0, 3.0, 0.0, 0.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 3.0],
            ],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = quad_to_cubic(types, coords)

    assert out_types[1, 0].item() == CommandType.CURVE_TO.value
    assert torch.allclose(
        out_coords[1, 0],
        torch.tensor([0.0, 2.0, 1.0, 3.0, 3.0, 3.0], dtype=torch.float32),
    )


def test_quad_to_cubic_can_merge_curves_in_same_transform() -> None:
    curve = _CUBIC_CURVES[0]
    types, coords = _cubic_segs_to_tensors([curve])
    q_types, q_coords = cubic_to_quad(types, coords)

    out_types, out_coords = quad_to_cubic(q_types, q_coords, merge_curves=True)

    _assert_single_cubic_matches(out_types, out_coords, curve)


def test_quad_to_cubic_merge_curves_runs_even_without_quadratics() -> None:
    types, coords = _line_path_to_tensors(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    )

    out_types, _out_coords = quad_to_cubic(types, coords, merge_curves=True)

    assert out_types.tolist().count(CommandType.LINE_TO.value) == 1


def test_cubic_to_quad_produces_quad_to_commands() -> None:
    curve = _CUBIC_CURVES[0]
    types, coords = _cubic_segs_to_tensors([curve])

    out_types, out_coords = cubic_to_quad(types, coords)

    assert CommandType.CURVE_TO.value not in out_types.tolist()
    assert CommandType.QUAD_TO.value in out_types.tolist()
    assert out_coords.shape[1] == 6


def test_cubic_to_quad_passes_through_non_cubic_commands() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = cubic_to_quad(types, coords)

    assert out_types.tolist() == types.tolist()
    assert torch.equal(out_coords, coords)


@pytest.mark.parametrize(
    "curve",
    [
        ((0.0, 0.0), (0.5, 1.0), (1.5, 1.0), (3.0, 0.0)),
        ((0.0, 0.0), (1.0 / 3.0, 0.0), (2.0 / 3.0, 0.0), (1.0, 0.0)),
    ],
)
def test_cubic_to_quad_exact_quadratics_use_one_segment_and_roundtrip(
    curve: _CubicSeg,
) -> None:
    # Cubics with p0 - 3*p1 + 3*p2 - p3 = 0 are exactly degree-elevated
    # quadratics; parallel endpoint tangents are covered by the straight case.
    types, coords = _cubic_segs_to_tensors([curve])

    q_types, q_coords = cubic_to_quad(types, coords)

    assert q_types.tolist().count(CommandType.QUAD_TO.value) == 1

    c_types, c_coords = quad_to_cubic(q_types, q_coords)
    m_types, m_coords = merge_curves(c_types, c_coords)

    _assert_single_cubic_matches(m_types, m_coords, curve, atol=1e-5)


@pytest.mark.parametrize("curve", _CUBIC_CURVES)
def test_cubic_to_quad_quad_to_cubic_merge_curves_roundtrip(curve: _CubicSeg) -> None:
    types, coords = _cubic_segs_to_tensors([curve])

    q_types, q_coords = cubic_to_quad(types, coords)
    c_types, c_coords = quad_to_cubic(q_types, q_coords)
    m_types, m_coords = merge_curves(c_types, c_coords)

    _assert_single_cubic_matches(m_types, m_coords, curve)


def test_cubic_to_quad_reports_unrepresentable_large_curve() -> None:
    # With normalized input this should be unusual, but public APIs should not
    # turn malformed/extreme input into a Rust panic.
    curve: _CubicSeg = (
        (0.0, 0.0),
        (0.0, 10_000.0),
        (10_000.0, 10_000.0),
        (10_000.0, 0.0),
    )
    types, coords = _cubic_segs_to_tensors([curve])

    with pytest.raises(ValueError, match="could not approximate"):
        cubic_to_quad(types, coords)


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_cubic_to_quad_reports_non_finite_curve(value: float) -> None:
    curve: _CubicSeg = (
        (0.0, 0.0),
        (value, 0.0),
        (1.0, 1.0),
        (1.0, 0.0),
    )
    types, coords = _cubic_segs_to_tensors([curve])

    with pytest.raises(ValueError, match="could not approximate"):
        cubic_to_quad(types, coords)


@pytest.mark.parametrize("curve", _CUBIC_CURVES)
def test_cubic_to_quad_quad_to_cubic_endpoints_are_exact(curve: _CubicSeg) -> None:
    types, coords = _cubic_segs_to_tensors([curve])

    q_types, q_coords = cubic_to_quad(types, coords)

    quad_indices = [
        i for i, t in enumerate(q_types.tolist()) if t == CommandType.QUAD_TO.value
    ]
    assert len(quad_indices) >= 1

    last_idx = quad_indices[-1]
    orig_end = torch.tensor([curve[3][0], curve[3][1]], dtype=torch.float32)
    assert torch.allclose(q_coords[last_idx, 4:6], orig_end, atol=1e-5)

    # For multi-segment results, intermediate on-curve points are midpoints of
    # adjacent off-curve control points (TrueType spline convention).
    if len(quad_indices) > 1:
        for k in range(len(quad_indices) - 1):
            qi = quad_indices[k]
            qi1 = quad_indices[k + 1]
            mid = (q_coords[qi, :2] + q_coords[qi1, :2]) / 2.0
            assert torch.allclose(q_coords[qi, 4:6], mid, atol=1e-5)


@pytest.mark.parametrize("curve", _CUBIC_CURVES)
def test_cubic_to_quad_implied_cubics_close_to_sub_cubics(curve: _CubicSeg) -> None:
    # cubic_to_quad splits the cubic into N quads at t = i/N.
    # quad_to_cubic converts each back; the resulting cp1/cp2 must be close to
    # the De Casteljau sub-cubic's own control points.
    # This directly validates the algorithm's tolerance guarantee.
    types, coords = _cubic_segs_to_tensors([curve])

    q_types, q_coords = cubic_to_quad(types, coords)
    c_types, c_coords = quad_to_cubic(q_types, q_coords)

    quad_count = q_types.tolist().count(CommandType.QUAD_TO.value)
    curve_indices = [
        i for i, t in enumerate(c_types.tolist()) if t == CommandType.CURVE_TO.value
    ]
    assert len(curve_indices) == quad_count

    for i, ci in enumerate(curve_indices):
        sub = _sub_cubic(curve, i / quad_count, (i + 1) / quad_count)
        actual_cp1 = (c_coords[ci, 0].item(), c_coords[ci, 1].item())
        actual_cp2 = (c_coords[ci, 2].item(), c_coords[ci, 3].item())
        err1 = math.sqrt(
            (actual_cp1[0] - sub[1][0]) ** 2 + (actual_cp1[1] - sub[1][1]) ** 2
        )
        err2 = math.sqrt(
            (actual_cp2[0] - sub[2][0]) ** 2 + (actual_cp2[1] - sub[2][1]) ** 2
        )
        assert err1 <= 2e-3, f"segment {i}: cp1 error {err1}"
        assert err2 <= 2e-3, f"segment {i}: cp2 error {err2}"


def test_remove_overlaps_merges_overlapping_contours() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 2.0, 0.0],
            [0, 0, 0, 0, 2.0, 2.0],
            [0, 0, 0, 0, 0.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 1.0, 0.0],
            [0, 0, 0, 0, 3.0, 0.0],
            [0, 0, 0, 0, 3.0, 2.0],
            [0, 0, 0, 0, 1.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = remove_overlaps(types, coords)

    assert out_types[-1].item() == CommandType.END.value
    assert out_types.tolist().count(CommandType.MOVE_TO.value) == 1
    assert out_types.tolist().count(CommandType.CLOSE.value) == 1
    expected = torch.tensor([0.0, 0.0, 3.0, 2.0])
    actual = torch.tensor(
        [
            out_coords[:, 4].min(),
            out_coords[:, 5].min(),
            out_coords[:, 4].max(),
            out_coords[:, 5].max(),
        ]
    )
    assert torch.allclose(actual, expected)


def test_patchify_reshapes_exact_multiple() -> None:
    types = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    coords = torch.zeros(4, 6, dtype=torch.float32)

    out_types, out_coords = patchify(types, coords, patch_size=2)

    assert out_types.shape == (2, 2)
    assert out_coords.shape == (2, 2, 6)
    assert torch.equal(out_types, types.view(2, 2))


def test_patchify_pads_when_not_exact_multiple() -> None:
    types = torch.tensor([1, 2, 3], dtype=torch.long)
    coords = torch.zeros(3, 6, dtype=torch.float32)

    out_types, out_coords = patchify(types, coords, patch_size=2)

    assert out_types.shape == (2, 2)
    assert out_coords.shape == (2, 2, 6)
    assert out_types[1, 1].item() == 0


def test_patchify_raises_for_invalid_patch_size() -> None:
    types = torch.tensor([1], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="patch_size must be >= 1"):
        patchify(types, coords, patch_size=0)


def test_render_bitmap_supports_coordinate_mapping_modes() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.75, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.75, 0.50],
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.50],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    fixed = _occupied_size(render_bitmap(types, coords, size=64, mode="fixed"))
    bbox = _occupied_size(render_bitmap(types, coords, size=64, mode="bbox"))
    bbox_square = _occupied_size(
        render_bitmap(types, coords, size=64, mode="bbox_square")
    )
    default = render_bitmap(types, coords, size=64)

    assert fixed == (22, 11)
    assert bbox == (22, 11)
    assert bbox_square == (64, 32)
    assert torch.equal(
        default, render_bitmap(types, coords, size=64, mode="bbox_square")
    )


def test_render_bitmap_bbox_returns_variable_size() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.10, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.60, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.60, 0.35],
            [0.0, 0.0, 0.0, 0.0, 0.10, 0.35],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    bitmap = render_bitmap(types, coords, size=64, mode="bbox")

    assert bitmap.shape == (11, 22)
    assert bitmap.device.type == "cpu"


def test_render_bitmap_rejects_unknown_mode() -> None:
    types = torch.tensor([CommandType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="mode must be one of"):
        render_bitmap(types, coords, mode="unknown")  # ty: ignore[invalid-argument-type]


def test_render_bitmap_bbox_empty_outline_returns_empty_bitmap() -> None:
    types = torch.tensor([CommandType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    bitmap = render_bitmap(types, coords, mode="bbox")

    assert bitmap.shape == (0, 0)


def test_render_bitmap_bbox_rejects_oversized_output() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 200.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 200.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="bbox output dimensions"):
        render_bitmap(types, coords, mode="bbox")


def _occupied_size(bitmap: torch.Tensor) -> tuple[int, int]:
    ys, xs = torch.nonzero(bitmap > 0, as_tuple=True)
    width = int(xs.max().item()) - int(xs.min().item()) + 1
    height = int(ys.max().item()) - int(ys.min().item()) + 1
    return width, height
