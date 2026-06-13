from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import vertical_flip


def _signed_area(types: torch.Tensor, coords: torch.Tensor) -> float:
    points = coords[
        (types == ElementType.MOVE_TO.value) | (types == ElementType.LINE_TO.value), 4:
    ]
    shifted = torch.roll(points, shifts=-1, dims=0)
    return float(
        (points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]).sum() / 2
    )


def test_vertical_flip_mirrors_y_around_bbox_center(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = vertical_flip(types, coords, preserve_winding=False)

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


def test_vertical_flip_preserves_closed_subpath_winding_by_default(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, out = vertical_flip(types, coords)
    assert _signed_area(out_types, out) == pytest.approx(_signed_area(types, coords))


def test_vertical_flip_can_leave_reflected_winding(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, out = vertical_flip(types, coords, preserve_winding=False)
    assert _signed_area(out_types, out) == pytest.approx(-_signed_area(types, coords))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_vertical_flip_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = (tensor.cuda() for tensor in simple_outline)

    out_types, out_coords = vertical_flip(types, coords)

    assert out_types.device.type == "cuda"
    assert out_coords.device.type == "cuda"
