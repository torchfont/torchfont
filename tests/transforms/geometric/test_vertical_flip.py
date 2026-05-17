import pytest
import torch

from torchfont.io import CommandType
from torchfont.transforms import vertical_flip

from ._helpers import _close_end_zeros, _simple_outline


def test_vertical_flip_mirrors_y_around_bbox_center() -> None:
    types, coords = _simple_outline()
    _, out = vertical_flip(types, coords)

    line_idx = types.tolist().index(CommandType.LINE_TO.value)
    assert out[line_idx, 5].item() == pytest.approx(1.0 - coords[line_idx, 5].item())


def test_vertical_flip_twice_is_identity() -> None:
    types, coords = _simple_outline()
    _, once = vertical_flip(types, coords)
    _, twice = vertical_flip(types, once)
    assert torch.allclose(twice, coords)


def test_vertical_flip_does_not_alter_x_coordinates() -> None:
    types, coords = _simple_outline()
    _, out = vertical_flip(types, coords)
    assert torch.allclose(out[:, 0], coords[:, 0])
    assert torch.allclose(out[:, 2], coords[:, 2])
    assert torch.allclose(out[:, 4], coords[:, 4])


def test_vertical_flip_preserves_padding_zeros() -> None:
    types, coords = _simple_outline()
    _, out = vertical_flip(types, coords)
    assert _close_end_zeros(types, out)
