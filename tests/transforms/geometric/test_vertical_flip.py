from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import vertical_flip


def test_vertical_flip_mirrors_y_around_bbox_center(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = vertical_flip(types, coords)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert out[line_idx, 5].item() == pytest.approx(1.0 - coords[line_idx, 5].item())


def test_vertical_flip_twice_is_identity(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, once = vertical_flip(types, coords)
    _, twice = vertical_flip(types, once)
    assert torch.allclose(twice, coords)


def test_vertical_flip_does_not_alter_x_coordinates(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = vertical_flip(types, coords)
    assert torch.allclose(out[:, 0], coords[:, 0])
    assert torch.allclose(out[:, 2], coords[:, 2])
    assert torch.allclose(out[:, 4], coords[:, 4])


def test_vertical_flip_preserves_padding_zeros(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    _, out = vertical_flip(types, coords)
    assert close_end_zeros(types, out)
