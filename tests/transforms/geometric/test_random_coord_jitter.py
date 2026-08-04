from collections.abc import Callable

import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import RandomCoordJitter
from torchfont.transforms.functional import coord_jitter


def test_random_coord_jitter_changes_only_active_coordinates(
    quad_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = quad_outline
    output = RandomCoordJitter(1.0)(Outline(types, coords))
    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert not torch.equal(output.coords[quad_idx, 0:2], coords[quad_idx, 0:2])
    assert output.coords[quad_idx, 2:4].tolist() == [0.0, 0.0]
    assert close_end_zeros(types, output.coords)


def test_random_coord_jitter_zero_std_is_identity(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    output = RandomCoordJitter(0.0)(outline)
    assert torch.equal(output.coords, outline.coords)


def test_random_coord_jitter_shares_noise_between_outlines(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    first, second = RandomCoordJitter(0.1)([outline, outline])
    assert torch.equal(first.coords, second.coords)


@pytest.mark.parametrize("std", [-0.1, float("nan"), float("inf")])
def test_random_coord_jitter_rejects_invalid_std(std: float) -> None:
    with pytest.raises(ValueError, match="std must be non-negative and finite"):
        RandomCoordJitter(std)


def test_coord_jitter_accepts_noise_for_a_longer_outline(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    noise = torch.ones((len(outline.types) + 1, 3, 2))

    output = coord_jitter(outline, noise)

    assert output.coords.shape == outline.coords.shape


@pytest.mark.parametrize("shape", [(1, 3, 2), (6, 6), (6, 2, 3)])
def test_coord_jitter_rejects_incompatible_noise_shape(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="noise must"):
        coord_jitter(Outline(*simple_outline), torch.ones(shape))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_coord_jitter_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output = RandomCoordJitter(0.1)(
        Outline(*(tensor.cuda() for tensor in simple_outline))
    )
    assert output.types.device.type == "cuda"
    assert output.coords.device.type == "cuda"
