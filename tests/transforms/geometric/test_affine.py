import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import affine

from ._helpers import (
    _close_end_zeros,
    _cubic_outline,
    _quad_outline,
    _simple_outline,
)


def test_affine_identity_leaves_coords_unchanged() -> None:
    types, coords = _simple_outline()
    _, out = affine(types, coords, angle=0.0, scale=1.0, shear=0.0)
    assert torch.allclose(out, coords, atol=1e-6)


def test_affine_does_not_modify_types() -> None:
    types, coords = _simple_outline()
    out_types, _ = affine(types, coords, angle=10.0)
    assert torch.equal(out_types, types)


def test_affine_preserves_padding_zeros() -> None:
    types, coords = _simple_outline()
    _, out = affine(types, coords, angle=45.0, scale=1.2, shear=5.0)
    assert _close_end_zeros(types, out)


def test_affine_90_degree_rotation() -> None:
    types, coords = _simple_outline()
    _, out = affine(types, coords, angle=90.0)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    ox, oy = coords[line_idx, 4].item(), coords[line_idx, 5].item()
    nx, ny = out[line_idx, 4].item(), out[line_idx, 5].item()
    assert nx == pytest.approx(1.0 - oy, abs=1e-5)
    assert ny == pytest.approx(ox, abs=1e-5)


def test_affine_scale_multiplies_coordinates() -> None:
    types, coords = _simple_outline()
    _, out = affine(types, coords, scale=2.0)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert out[line_idx, 4].item() == pytest.approx(
        coords[line_idx, 4].item() * 2.0 - 0.5, abs=1e-5
    )
    assert out[line_idx, 5].item() == pytest.approx(
        coords[line_idx, 5].item() * 2.0 - 0.5, abs=1e-5
    )


def test_affine_translate_shifts_endpoints() -> None:
    types, coords = _simple_outline()
    _, out = affine(types, coords, translate=(0.1, 0.2))

    move_idx = types.tolist().index(ElementType.MOVE_TO.value)
    assert out[move_idx, 4].item() == pytest.approx(
        coords[move_idx, 4].item() + 0.1, abs=1e-5
    )
    assert out[move_idx, 5].item() == pytest.approx(
        coords[move_idx, 5].item() + 0.2, abs=1e-5
    )


def test_affine_transforms_all_cubic_pairs() -> None:
    types, coords = _cubic_outline()
    _, out = affine(types, coords, angle=90.0)

    curve_idx = types.tolist().index(ElementType.CURVE_TO.value)
    assert not torch.allclose(out[curve_idx, 0:2], coords[curve_idx, 0:2])
    assert not torch.allclose(out[curve_idx, 2:4], coords[curve_idx, 2:4])
    assert not torch.allclose(out[curve_idx, 4:6], coords[curve_idx, 4:6])


def test_affine_quad_pair1_stays_zero() -> None:
    types, coords = _quad_outline()
    _, out = affine(types, coords, angle=45.0, translate=(0.05, 0.05))

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


def test_affine_non_positive_scale_raises() -> None:
    types, coords = _simple_outline()
    with pytest.raises(ValueError, match="scale must be positive"):
        affine(types, coords, scale=0.0)
