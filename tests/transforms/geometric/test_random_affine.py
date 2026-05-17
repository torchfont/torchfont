from __future__ import annotations

from typing import TYPE_CHECKING

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import random_affine

if TYPE_CHECKING:
    from collections.abc import Callable


def test_random_affine_deterministic_with_generator(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g1 = torch.Generator().manual_seed(7)
    g2 = torch.Generator().manual_seed(7)
    _, out1 = random_affine(types, coords, degrees=30.0, generator=g1)
    _, out2 = random_affine(types, coords, degrees=30.0, generator=g2)
    assert torch.allclose(out1, out2)


def test_random_affine_preserves_padding_zeros(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(1)
    _, out = random_affine(
        types,
        coords,
        degrees=15.0,
        translate=(0.05, 0.05),
        scale=(0.9, 1.1),
        generator=g,
    )
    assert close_end_zeros(types, out)


def test_random_affine_does_not_modify_types(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    out_types, _ = random_affine(types, coords, degrees=10.0)
    assert torch.equal(out_types, types)


def test_random_affine_translation_within_bounds(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(3)
    move_idx = types.tolist().index(ElementType.MOVE_TO.value)
    _, out = random_affine(types, coords, translate=(0.1, 0.2), generator=g)
    dx = abs(out[move_idx, 4].item() - coords[move_idx, 4].item())
    dy = abs(out[move_idx, 5].item() - coords[move_idx, 5].item())
    assert dx <= 0.1 + 1e-6
    assert dy <= 0.2 + 1e-6


def test_random_affine_invalid_scale_raises(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="positive"):
        random_affine(types, coords, scale=(-1.0, 1.0))


def test_random_affine_reversed_scale_range_raises(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="min <= max"):
        random_affine(types, coords, scale=(2.0, 1.0))


def test_random_affine_quad_pair1_stays_zero(
    quad_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = quad_outline
    g = torch.Generator().manual_seed(5)
    _, out = random_affine(
        types, coords, degrees=45.0, translate=(0.05, 0.05), generator=g
    )

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)
