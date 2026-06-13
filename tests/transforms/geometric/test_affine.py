from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import affine


def test_affine_identity_leaves_coords_unchanged(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = affine(types, coords, angle=0.0, scale=1.0, shear=0.0)
    assert torch.allclose(out, coords, atol=1e-6)


def test_affine_does_not_modify_types(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, _ = affine(types, coords, angle=10.0)
    assert torch.equal(out_types, types)


def test_affine_preserves_padding_zeros(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    _, out = affine(types, coords, angle=45.0, scale=1.2, shear=5.0)
    assert close_end_zeros(types, out)


def test_affine_90_degree_rotation(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = affine(types, coords, angle=90.0)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    ox, oy = coords[line_idx, 4].item(), coords[line_idx, 5].item()
    nx, ny = out[line_idx, 4].item(), out[line_idx, 5].item()
    assert nx == pytest.approx(1.0 - oy, abs=1e-5)
    assert ny == pytest.approx(ox, abs=1e-5)


def test_affine_scale_multiplies_coordinates(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = affine(types, coords, scale=2.0)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert out[line_idx, 4].item() == pytest.approx(
        coords[line_idx, 4].item() * 2.0 - 0.5, abs=1e-5
    )
    assert out[line_idx, 5].item() == pytest.approx(
        coords[line_idx, 5].item() * 2.0 - 0.5, abs=1e-5
    )


def test_affine_translate_shifts_endpoints(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = affine(types, coords, translate=(0.1, 0.2))

    move_idx = types.tolist().index(ElementType.MOVE_TO.value)
    assert out[move_idx, 4].item() == pytest.approx(
        coords[move_idx, 4].item() + 0.1, abs=1e-5
    )
    assert out[move_idx, 5].item() == pytest.approx(
        coords[move_idx, 5].item() + 0.2, abs=1e-5
    )


def test_affine_transforms_all_cubic_pairs(
    cubic_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = cubic_outline
    _, out = affine(types, coords, angle=90.0)

    curve_idx = types.tolist().index(ElementType.CURVE_TO.value)
    assert not torch.allclose(out[curve_idx, 0:2], coords[curve_idx, 0:2])
    assert not torch.allclose(out[curve_idx, 2:4], coords[curve_idx, 2:4])
    assert not torch.allclose(out[curve_idx, 4:6], coords[curve_idx, 4:6])


def test_affine_quad_pair1_stays_zero(
    quad_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = quad_outline
    _, out = affine(types, coords, angle=45.0, translate=(0.05, 0.05))

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_affine_invalid_scale_raises(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    scale: float,
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="scale must be positive and finite"):
        affine(types, coords, scale=scale)


@pytest.mark.parametrize("angle", [float("nan"), float("inf")])
def test_affine_rejects_non_finite_angle(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    angle: float,
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="angle must be finite"):
        affine(types, coords, angle=angle)


@pytest.mark.parametrize("shear", [float("nan"), float("inf")])
def test_affine_rejects_non_finite_shear(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    shear: float,
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="shear must be finite"):
        affine(types, coords, shear=shear)


@pytest.mark.parametrize(
    "translate",
    [(float("nan"), 0.0), (0.0, float("inf"))],
)
def test_affine_rejects_non_finite_translate(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    translate: tuple[float, float],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="translate values must be finite"):
        affine(types, coords, translate=translate)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_affine_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = (tensor.cuda() for tensor in simple_outline)

    out_types, out_coords = affine(types, coords, angle=15.0)

    assert out_types.device.type == "cuda"
    assert out_coords.device.type == "cuda"
