from collections.abc import Callable

import pytest
import torch

from torchfont.transforms import Outline, RandomAffine


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


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_affine_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output = RandomAffine(degrees=45.0)(
        Outline(*(tensor.cuda() for tensor in simple_outline))
    )
    assert output.types.device.type == "cuda"
    assert output.coords.device.type == "cuda"
