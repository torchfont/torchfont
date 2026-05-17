import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import horizontal_flip

from ._helpers import (
    _close_end_zeros,
    _cubic_outline,
    _quad_outline,
    _simple_outline,
)


def test_horizontal_flip_mirrors_x_around_bbox_center() -> None:
    types, coords = _simple_outline()
    _, out = horizontal_flip(types, coords)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert out[line_idx, 4].item() == pytest.approx(1.0 - coords[line_idx, 4].item())


def test_horizontal_flip_twice_is_identity() -> None:
    types, coords = _simple_outline()
    _, once = horizontal_flip(types, coords)
    _, twice = horizontal_flip(types, once)
    assert torch.allclose(twice, coords)


def test_horizontal_flip_does_not_modify_types() -> None:
    types, coords = _simple_outline()
    out_types, _ = horizontal_flip(types, coords)
    assert torch.equal(out_types, types)


def test_horizontal_flip_preserves_padding_zeros() -> None:
    types, coords = _simple_outline()
    _, out = horizontal_flip(types, coords)
    assert _close_end_zeros(types, out)


def test_horizontal_flip_transforms_cubic_control_points() -> None:
    types, coords = _cubic_outline()
    _, out = horizontal_flip(types, coords)

    curve_idx = types.tolist().index(ElementType.CURVE_TO.value)
    assert out[curve_idx, 0].item() == pytest.approx(1.0 - coords[curve_idx, 0].item())
    assert out[curve_idx, 2].item() == pytest.approx(1.0 - coords[curve_idx, 2].item())


def test_horizontal_flip_quad_pair1_stays_zero() -> None:
    types, coords = _quad_outline()
    _, out = horizontal_flip(types, coords)

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


def test_horizontal_flip_does_not_alter_y_coordinates() -> None:
    types, coords = _simple_outline()
    _, out = horizontal_flip(types, coords)
    assert torch.allclose(out[:, 1], coords[:, 1])
    assert torch.allclose(out[:, 3], coords[:, 3])
    assert torch.allclose(out[:, 5], coords[:, 5])
