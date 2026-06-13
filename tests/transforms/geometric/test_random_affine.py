from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import random_affine


def test_random_affine_deterministic_with_generator(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    _, out1 = random_affine(types, coords, degrees=30.0, generator=g1)
    _, out2 = random_affine(types, coords, degrees=30.0, generator=g2)
    assert torch.allclose(out1, out2)


def test_random_affine_preserves_padding_zeros(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(1)
    _, out = random_affine(
        types,
        coords,
        degrees=15.0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        generator=g,
    )
    assert close_end_zeros(types, out)


def test_random_affine_does_not_modify_types(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, _ = random_affine(types, coords, degrees=10.0)
    assert torch.equal(out_types, types)


def test_random_affine_translation_within_bounds(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(3)
    move_idx = types.tolist().index(ElementType.MOVE_TO.value)
    _, out = random_affine(types, coords, translate=(0.1, 0.2), generator=g)
    dx = abs(out[move_idx, 4].item() - coords[move_idx, 4].item())
    dy = abs(out[move_idx, 5].item() - coords[move_idx, 5].item())
    assert dx <= 0.1 + 1e-6
    assert dy <= 0.2 + 1e-6


@pytest.mark.parametrize(
    "scale",
    [(-1.0, 1.0), (float("nan"), 1.0), (1.0, float("inf"))],
)
def test_random_affine_invalid_scale_raises(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    scale: tuple[float, float],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="positive and finite"):
        random_affine(types, coords, scale=scale)


def test_random_affine_reversed_scale_range_raises(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="min <= max"):
        random_affine(types, coords, scale=(2.0, 1.0))


@pytest.mark.parametrize(
    "degrees",
    [float("nan"), (0.0, float("inf"))],
)
def test_random_affine_rejects_non_finite_degrees(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    degrees: float | tuple[float, float],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="range values must be finite"):
        random_affine(types, coords, degrees=degrees)


@pytest.mark.parametrize(
    "shear",
    [float("inf"), (float("nan"), 0.0)],
)
def test_random_affine_rejects_non_finite_shear(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    shear: float | tuple[float, float],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="range values must be finite"):
        random_affine(types, coords, shear=shear)


@pytest.mark.parametrize(
    "translate",
    [(float("inf"), 0.0), (0.0, float("nan"))],
)
def test_random_affine_rejects_non_finite_translate(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    translate: tuple[float, float],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="translate values must be finite"):
        random_affine(types, coords, translate=translate)


def test_random_affine_quad_pair1_stays_zero(
    quad_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = quad_outline
    g = torch.Generator().manual_seed(5)
    _, out = random_affine(
        types, coords, degrees=45.0, translate=(0.05, 0.05), generator=g
    )

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_affine_accepts_cpu_generator_for_cuda_input(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = (tensor.cuda() for tensor in simple_outline)
    generator = torch.Generator().manual_seed(5)

    out_types, out_coords = random_affine(
        types,
        coords,
        degrees=45.0,
        generator=generator,
    )

    assert out_types.device.type == "cuda"
    assert out_coords.device.type == "cuda"
