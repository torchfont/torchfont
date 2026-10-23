import math

import pytest
import torch

from tests._pairs import cubic_to_quad, merge_curves, quad_to_cubic
from torchfont import ElementType

from ._helpers import (
    _CUBIC_CURVES,
    _assert_single_cubic_matches,
    _cubic_segs_to_tensors,
    _CubicSeg,
)


def test_cubic_to_quad_produces_quad_to_commands() -> None:
    curve = _CUBIC_CURVES[0]
    types, coords = _cubic_segs_to_tensors([curve])

    out_types, out_coords = cubic_to_quad(types, coords)

    assert ElementType.CURVE_TO.value not in out_types.tolist()
    assert ElementType.QUAD_TO.value in out_types.tolist()
    assert out_coords.shape[1] == 6


def test_cubic_to_quad_passes_through_non_cubic_commands() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
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
    types, coords = _cubic_segs_to_tensors([curve])

    q_types, q_coords = cubic_to_quad(types, coords)

    assert q_types.tolist().count(ElementType.QUAD_TO.value) == 1

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
def test_cubic_to_quad_endpoint_is_exact(curve: _CubicSeg) -> None:
    types, coords = _cubic_segs_to_tensors([curve])

    q_types, q_coords = cubic_to_quad(types, coords)

    quad_indices = [
        i for i, t in enumerate(q_types.tolist()) if t == ElementType.QUAD_TO.value
    ]
    assert len(quad_indices) >= 1

    last_idx = quad_indices[-1]
    orig_end = torch.tensor([curve[3][0], curve[3][1]], dtype=torch.float32)
    assert torch.allclose(q_coords[last_idx, 4:6], orig_end, atol=1e-5)
