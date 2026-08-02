import pytest
import torch

from torchfont.structures import Outline
from torchfont.transforms import RandomHorizontalFlip


@pytest.mark.parametrize(("p", "changes"), [(0.0, False), (1.0, True)])
def test_random_horizontal_flip_probability_boundaries(
    simple_outline: tuple[torch.Tensor, torch.Tensor], p: float, *, changes: bool
) -> None:
    types, coords = simple_outline
    output = RandomHorizontalFlip(p)(Outline(types, coords))
    assert (not torch.equal(output.coords, coords)) is changes


def test_random_horizontal_flip_shares_decision_between_outlines(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*simple_outline)
    first, second = RandomHorizontalFlip()([outline, outline])
    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan")])
def test_random_horizontal_flip_rejects_invalid_probability(p: float) -> None:
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        RandomHorizontalFlip(p)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_horizontal_flip_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output = RandomHorizontalFlip(1.0)(
        Outline(*(tensor.cuda() for tensor in simple_outline))
    )
    assert output.types.device.type == "cuda"
    assert output.coords.device.type == "cuda"
