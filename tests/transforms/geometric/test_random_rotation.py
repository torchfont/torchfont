from collections.abc import Callable

import pytest
import torch

from torchfont import Outline
from torchfont.transforms import RandomRotation
from torchfont.transforms import functional as F  # noqa: N812


def test_rotate_matches_affine(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    assert torch.equal(
        F.rotate(outline, 15.0).coords, F.affine(outline, angle=15.0).coords
    )


def test_random_rotation_is_reproducible_with_torch_seed(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    torch.manual_seed(7)
    first = RandomRotation(30.0)(outline)
    torch.manual_seed(7)
    second = RandomRotation(30.0)(outline)
    assert torch.equal(first.coords, second.coords)


def test_random_rotation_shares_angle_between_outlines(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    outline = Outline(*simple_outline)
    first, second = RandomRotation((-10.0, 20.0))([outline, outline])
    assert torch.equal(first.coords, second.coords)
    assert close_end_zeros(first.types, first.coords)


def test_random_rotation_samples_requested_range() -> None:
    transform = RandomRotation((10.0, 20.0))

    angle = transform.make_params([])["angle"]

    assert 10.0 <= angle <= 20.0


def test_random_rotation_number_creates_symmetric_range() -> None:
    assert RandomRotation(15.0).degrees == (-15.0, 15.0)


@pytest.mark.parametrize("degrees", [float("nan"), (0.0, float("inf")), (10.0, -10.0)])
def test_random_rotation_rejects_invalid_degrees(
    degrees: float | tuple[float, float],
) -> None:
    with pytest.raises(ValueError, match="range values must be finite and ordered"):
        RandomRotation(degrees)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_rotation_rejects_cuda_input(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*(tensor.cuda() for tensor in simple_outline))
    with pytest.raises(NotImplementedError, match="CUDA"):
        RandomRotation(10.0)(outline)
