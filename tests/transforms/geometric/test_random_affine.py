from collections.abc import Callable

import pytest
import torch

from torchfont import Outline
from torchfont.transforms import RandomAffine


def test_random_affine_is_reproducible_with_torch_seed(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    torch.manual_seed(7)
    first = RandomAffine(degrees=30.0)(outline)
    torch.manual_seed(7)
    second = RandomAffine(degrees=30.0)(outline)
    assert torch.equal(first.coords, second.coords)


def test_random_affine_preserves_padding_and_shares_parameters(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    outline = Outline(*simple_outline)
    first, second = RandomAffine(
        degrees=15.0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
    )([outline, outline])
    assert torch.equal(first.coords, second.coords)
    assert close_end_zeros(first.types, first.coords)


def test_random_affine_samples_x_and_y_shear_ranges() -> None:
    transform = RandomAffine(shear=(-20.0, -10.0, 10.0, 20.0))

    shear_x, shear_y = transform.make_params([])["shear"]

    assert -20.0 <= shear_x <= -10.0
    assert 10.0 <= shear_y <= 20.0


def test_random_affine_two_value_shear_only_samples_x() -> None:
    transform = RandomAffine(shear=(-20.0, -10.0))

    shear_x, shear_y = transform.make_params([])["shear"]

    assert -20.0 <= shear_x <= -10.0
    assert shear_y == 0.0


@pytest.mark.parametrize(
    "scale",
    [(-1.0, 1.0), (float("nan"), 1.0), (1.0, float("inf")), (2.0, 1.0)],
)
def test_random_affine_rejects_invalid_scale(scale: tuple[float, float]) -> None:
    with pytest.raises(ValueError, match="scale values"):
        RandomAffine(scale=scale)


@pytest.mark.parametrize(
    "degrees",
    [float("nan"), (0.0, float("inf")), (10.0, -10.0)],
)
def test_random_affine_rejects_invalid_degrees(
    degrees: float | tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="range values must be finite and ordered"):
        RandomAffine(degrees=degrees)


@pytest.mark.parametrize("name", ["degrees", "translate", "scale"])
def test_random_affine_rejects_non_pair_ranges(name: str) -> None:
    with pytest.raises(ValueError, match="too many values to unpack"):
        RandomAffine(**{name: (1.0, 2.0, 3.0)})  # ty: ignore[invalid-argument-type]


def test_random_affine_rejects_invalid_shear_length() -> None:
    with pytest.raises(ValueError, match="two or four values"):
        RandomAffine(shear=(1.0, 2.0, 3.0))  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize(
    "shear",
    [(0.0, float("nan"), 0.0, 1.0), (0.0, 1.0, 2.0, float("inf"))],
)
def test_random_affine_rejects_invalid_four_value_shear(
    shear: tuple[float, float, float, float],
) -> None:
    with pytest.raises(ValueError, match="range values must be finite and ordered"):
        RandomAffine(shear=shear)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_affine_rejects_cuda_input(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    with pytest.raises(NotImplementedError, match="CUDA"):
        RandomAffine(degrees=45.0)(
            Outline(*(tensor.cuda() for tensor in simple_outline))
        )
