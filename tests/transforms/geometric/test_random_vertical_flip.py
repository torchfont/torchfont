import torch

from torchfont.transforms import random_vertical_flip

from ._helpers import _simple_outline


def test_random_vertical_flip_applies_with_p1() -> None:
    types, coords = _simple_outline()
    g = torch.Generator().manual_seed(0)
    _, out = random_vertical_flip(types, coords, p=1.0, generator=g)
    assert not torch.equal(out, coords)


def test_random_vertical_flip_skips_with_p0() -> None:
    types, coords = _simple_outline()
    _, out = random_vertical_flip(types, coords, p=0.0)
    assert torch.equal(out, coords)
