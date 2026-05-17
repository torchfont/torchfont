import torch

from torchfont.transforms import random_horizontal_flip

from ._helpers import _simple_outline


def test_random_horizontal_flip_applies_with_p1() -> None:
    types, coords = _simple_outline()
    g = torch.Generator().manual_seed(0)
    _, out = random_horizontal_flip(types, coords, p=1.0, generator=g)
    assert not torch.equal(out, coords)


def test_random_horizontal_flip_skips_with_p0() -> None:
    types, coords = _simple_outline()
    _, out = random_horizontal_flip(types, coords, p=0.0)
    assert torch.equal(out, coords)


def test_random_horizontal_flip_deterministic_with_generator() -> None:
    types, coords = _simple_outline()
    g1 = torch.Generator().manual_seed(42)
    g2 = torch.Generator().manual_seed(42)
    _, out1 = random_horizontal_flip(types, coords, generator=g1)
    _, out2 = random_horizontal_flip(types, coords, generator=g2)
    assert torch.equal(out1, out2)
