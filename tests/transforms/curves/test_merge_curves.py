from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import cubic_to_quad, merge_curves

from ._helpers import (
    _CUBIC_CURVES,
    _assert_single_cubic_matches,
    _cubic_segs_to_tensors,
    _CubicSeg,
    _line_path_to_tensors,
    _Point,
    _quad_segs_to_tensors,
    _QuadSeg,
    _split_cubic,
    _split_quad,
)


def test_merge_curves_collinear_lines_are_merged() -> None:
    types, coords = _line_path_to_tensors(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    )

    out_types, out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(ElementType.LINE_TO.value) == 1
    line_idx = out_types.tolist().index(ElementType.LINE_TO.value)
    assert out_coords[line_idx, 4].item() == pytest.approx(3.0)
    assert out_coords[line_idx, 5].item() == pytest.approx(0.0)


def test_merge_curves_allows_normalization_roundoff_for_lines() -> None:
    upm = 1000.0
    points = [(x / upm, y / upm) for x, y in [(101, 37), (202, 74), (303, 111)]]
    types, coords = _line_path_to_tensors(points)

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(ElementType.LINE_TO.value) == 1


@pytest.mark.parametrize(
    "points",
    [
        pytest.param([(0.0, 0.0), (1.0, 0.0), (1.0, 1.0)], id="corner"),
        # Tiny angular bend, but far from the merged line in normalized coordinates.
        pytest.param(
            [(0.0, 0.0), (1_000.0, 0.0), (2_000.0, 0.5)], id="absolute-tolerance"
        ),
        pytest.param([(0.0, 0.0), (1.0, 1e-7), (2.0, 0.0)], id="tiny-bend"),
        pytest.param([(0.0, 0.0), (1.0, 0.0), (0.5, 0.0)], id="antiparallel"),
    ],
)
def test_merge_curves_non_mergeable_lines_stay_separate(points: list[_Point]) -> None:
    types, coords = _line_path_to_tensors(points)

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(ElementType.LINE_TO.value) == 2


def test_merge_curves_does_not_accumulate_line_error() -> None:
    # A one-unit perpendicular segment is within the absolute tolerance when
    # preceded by a long line, but must not turn the following polyline into a
    # single diagonal chord.
    upm = 1024.0
    points = [(87.0 / upm, 193.0 / upm), (87.0 / upm, 0.0)]
    points.extend((x / upm, -(x // 64) / upm) for x in range(88, 184))
    types, coords = _line_path_to_tensors(points)

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(ElementType.LINE_TO.value) > 1


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

    assert out_types.tolist().count(ElementType.QUAD_TO.value) == 1
    idx = out_types.tolist().index(ElementType.QUAD_TO.value)
    assert torch.allclose(
        out_coords[idx],
        torch.tensor([1.0, 2.0, 0.0, 0.0, 3.0, 0.0]),
        atol=1e-4,
    )


def test_merge_curves_incompatible_quads_are_not_merged() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.QUAD_TO.value,
            ElementType.QUAD_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
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

    assert out_types.tolist().count(ElementType.QUAD_TO.value) == 2


def test_merge_curves_incompatible_cubics_are_not_merged() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.CURVE_TO.value,
            ElementType.CURVE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
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

    assert out_types.tolist().count(ElementType.CURVE_TO.value) == 2


def test_merge_curves_preserves_structure_outside_subpath() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.zeros(4, 6, dtype=torch.float32)
    coords[1, 4] = 1.0

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types[-1].item() == ElementType.END.value
    assert ElementType.CLOSE.value in out_types.tolist()


def test_merge_curves_multiple_subpaths() -> None:
    def _subpath(ox: float) -> tuple[list[int], list[list[float]]]:
        t = [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
        ]
        c = [
            [0.0, 0.0, 0.0, 0.0, ox, 0.0],
            [0.0, 0.0, 0.0, 0.0, ox + 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, ox + 2.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
        return t, c

    t1, c1 = _subpath(0.0)
    t2, c2 = _subpath(10.0)
    end_t = [ElementType.END.value]
    end_c = [[0.0] * 6]

    types = torch.tensor(t1 + t2 + end_t, dtype=torch.long)
    coords = torch.tensor(c1 + c2 + end_c, dtype=torch.float32)

    out_types, _out_coords = merge_curves(types, coords)

    assert out_types.tolist().count(ElementType.LINE_TO.value) == 2
    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 2


@pytest.mark.parametrize("transform", [cubic_to_quad, merge_curves])
def test_variable_length_transforms_reject_mismatched_coords(
    transform: Callable[
        [torch.Tensor, torch.Tensor], tuple[torch.Tensor, torch.Tensor]
    ],
) -> None:
    types = torch.tensor(
        [ElementType.MOVE_TO.value, ElementType.CURVE_TO.value],
        dtype=torch.long,
    )
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="coords length"):
        transform(types, coords)
