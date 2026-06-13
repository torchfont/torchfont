from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import random_coord_jitter


def test_random_coord_jitter_changes_active_coords(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(42)
    _, out = random_coord_jitter(types, coords, std=0.1, generator=g)

    line_idx = types.tolist().index(ElementType.LINE_TO.value)
    assert not torch.equal(out[line_idx, 4:6], coords[line_idx, 4:6])


def test_random_coord_jitter_zero_std_is_identity(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    _, out = random_coord_jitter(types, coords, std=0.0)
    assert torch.equal(out, coords)


def test_random_coord_jitter_preserves_padding_zeros(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    close_end_zeros: Callable[[torch.Tensor, torch.Tensor], bool],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(1)
    _, out = random_coord_jitter(types, coords, std=1.0, generator=g)
    assert close_end_zeros(types, out)


def test_random_coord_jitter_quad_pair1_stays_zero(
    quad_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = quad_outline
    g = torch.Generator().manual_seed(2)
    _, out = random_coord_jitter(types, coords, std=1.0, generator=g)

    quad_idx = types.tolist().index(ElementType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


def test_random_coord_jitter_jitters_all_cubic_pairs(
    cubic_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = cubic_outline
    g = torch.Generator().manual_seed(7)
    _, out = random_coord_jitter(types, coords, std=0.5, generator=g)

    curve_idx = types.tolist().index(ElementType.CURVE_TO.value)
    assert not torch.equal(out[curve_idx, 0:2], coords[curve_idx, 0:2])
    assert not torch.equal(out[curve_idx, 2:4], coords[curve_idx, 2:4])
    assert not torch.equal(out[curve_idx, 4:6], coords[curve_idx, 4:6])


def test_random_coord_jitter_does_not_modify_types(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g = torch.Generator().manual_seed(0)
    out_types, _ = random_coord_jitter(types, coords, std=0.1, generator=g)
    assert torch.equal(out_types, types)


@pytest.mark.parametrize("std", [-0.1, float("nan"), float("inf")])
def test_random_coord_jitter_invalid_std_raises(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
    std: float,
) -> None:
    types, coords = simple_outline
    with pytest.raises(ValueError, match="std must be non-negative and finite"):
        random_coord_jitter(types, coords, std=std)


def test_random_coord_jitter_deterministic_with_generator(
    simple_outline: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = simple_outline
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    _, out1 = random_coord_jitter(types, coords, std=0.05, generator=g1)
    _, out2 = random_coord_jitter(types, coords, std=0.05, generator=g2)
    assert torch.equal(out1, out2)
