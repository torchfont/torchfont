import math

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import cubic_to_quad, merge_curves, quad_to_cubic

from ._helpers import (
    _CUBIC_CURVES,
    _assert_single_cubic_matches,
    _cubic_segs_to_tensors,
    _CubicSeg,
    _sub_cubic,
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
    # Cubics with p0 - 3*p1 + 3*p2 - p3 = 0 are exactly degree-elevated
    # quadratics; parallel endpoint tangents are covered by the straight case.
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
        i for i, t in enumerate(q_types.tolist()) if t == ElementType.QUAD_TO.value
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

    quad_count = q_types.tolist().count(ElementType.QUAD_TO.value)
    curve_indices = [
        i for i, t in enumerate(c_types.tolist()) if t == ElementType.CURVE_TO.value
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
