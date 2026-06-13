import pytest
import torch

from torchfont.transforms import random_horizontal_flip


def test_random_horizontal_flip_applies_with_p1(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(0)
    _, out = random_horizontal_flip(types, coords, p=1.0, generator=g)
    assert not torch.equal(out, coords)


def test_random_horizontal_flip_skips_with_p0(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = random_horizontal_flip(types, coords, p=0.0)
    assert torch.equal(out, coords)


def test_random_horizontal_flip_deterministic_with_generator(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, out1 = random_horizontal_flip(types, coords, generator=g1)
    _, out2 = random_horizontal_flip(types, coords, generator=g2)
    assert torch.equal(out1, out2)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available")
def test_random_horizontal_flip_accepts_cpu_generator_for_cuda_input(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = (tensor.cuda() for tensor in simple_outline)
    generator = torch.Generator().manual_seed(0)

    out_types, out_coords = random_horizontal_flip(
        types,
        coords,
        p=1.0,
        generator=generator,
    )

    assert out_types.device.type == "cuda"
    assert out_coords.device.type == "cuda"
