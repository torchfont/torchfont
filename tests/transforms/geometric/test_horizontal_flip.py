from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import horizontal_flip


def _signed_area(types: torch.Tensor, coords: torch.Tensor) -> float:
    points = coords[
        (types == ElementType.MOVE_TO.value) | (types == ElementType.LINE_TO.value), 4:
    ]
    shifted = torch.roll(points, shifts=-1, dims=0)
    return float(
        (points[:, 0] * shifted[:, 1] - shifted[:, 0] * points[:, 1]).sum() / 2
    )


def test_horizontal_flip_mirrors_x_around_bbox_center(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = horizontal_flip(types, coords)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert out[line_idx, 4].item() == pytest.approx(1.0 - coords[line_idx, 4].item())


def test_horizontal_flip_twice_is_identity(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, once = horizontal_flip(types, coords)
    _, twice = horizontal_flip(types, once)
    assert torch.allclose(twice, coords)


def test_horizontal_flip_does_not_modify_types(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, _ = horizontal_flip(types, coords)
    assert torch.equal(out_types, types)


def test_horizontal_flip_preserves_padding_zeros(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    _, out = horizontal_flip(types, coords)
    assert close_end_zeros(types, out)


def test_horizontal_flip_transforms_cubic_control_points(
    cubic_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = cubic_outline
    _, out = horizontal_flip(types, coords, preserve_winding=False)

    curve_idx = types.tolist().index(ElementType.CURVE_TO.value)
    assert out[curve_idx, 0].item() == pytest.approx(1.0 - coords[curve_idx, 0].item())
    assert out[curve_idx, 2].item() == pytest.approx(1.0 - coords[curve_idx, 2].item())


def test_horizontal_flip_quad_pair1_stays_zero(
    quad_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = quad_outline
    _, out = horizontal_flip(types, coords)

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


def test_horizontal_flip_does_not_alter_y_coordinates(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = horizontal_flip(types, coords, preserve_winding=False)
    assert torch.allclose(out[:, 1], coords[:, 1])
    assert torch.allclose(out[:, 3], coords[:, 3])
    assert torch.allclose(out[:, 5], coords[:, 5])


def test_horizontal_flip_preserves_closed_subpath_winding_by_default(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, out = horizontal_flip(types, coords)
    assert _signed_area(out_types, out) == pytest.approx(_signed_area(types, coords))


def test_horizontal_flip_can_leave_reflected_winding(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, out = horizontal_flip(types, coords, preserve_winding=False)
    assert _signed_area(out_types, out) == pytest.approx(-_signed_area(types, coords))


def test_horizontal_flip_does_not_reverse_open_subpaths(
    open_subpath: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = open_subpath
    out_types, out = horizontal_flip(types, coords)
    assert torch.equal(out_types, types)
    assert out[0, 4].item() == pytest.approx(1.0)
    assert out[1, 4].item() == pytest.approx(2.0)
