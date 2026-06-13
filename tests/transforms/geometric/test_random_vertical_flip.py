import pytest
import torch

from torchfont.transforms import random_vertical_flip


def test_random_vertical_flip_applies_with_p1(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(0)
    _, out = random_vertical_flip(
        types,
        coords,
        p=1.0,
        preserve_winding=False,
        generator=g,
    )
    assert not torch.equal(out, coords)


def test_random_vertical_flip_skips_with_p0(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = random_vertical_flip(types, coords, p=0.0)
    assert torch.equal(out, coords)


@pytest.mark.parametrize("p", [-0.1, 1.1, float("nan"), float("inf")])
def test_random_vertical_flip_rejects_invalid_probability(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    p: float,
) -> None:
    types, coords = simple_outline

    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        random_vertical_flip(types, coords, p=p)
