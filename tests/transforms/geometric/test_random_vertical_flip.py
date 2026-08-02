import pytest
import torch

from torchfont.structures import Outline
from torchfont.transforms import RandomVerticalFlip


@pytest.mark.parametrize(("p", "changes"), [(0.0, False), (1.0, True)])
def test_random_vertical_flip_probability_boundaries(
    simple_outline: tuple[torch.Tensor, torch.Tensor], p: float, *, changes: bool
) -> None:
    types, coords = simple_outline
    output = RandomVerticalFlip(p, preserve_winding=False)(Outline(types, coords))
    assert (not torch.equal(output.coords, coords)) is changes


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_vertical_flip_preserves_cuda_device(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output = RandomVerticalFlip(1.0)(
        Outline(*(tensor.cuda() for tensor in simple_outline))
    )
    assert output.types.device.type == "cuda"
    assert output.coords.device.type == "cuda"
