import pytest
import torch

from torchfont.io import CommandType
from torchfont.transforms import random_coord_jitter

from ._helpers import (
    _close_end_zeros,
    _cubic_outline,
    _quad_outline,
    _simple_outline,
)


def test_random_coord_jitter_changes_active_coords() -> None:
    types, coords = _simple_outline()
    g = torch.Generator().manual_seed(42)
    _, out = random_coord_jitter(types, coords, std=0.1, generator=g)

    line_idx = types.tolist().index(CommandType.LINE_TO.value)
    assert not torch.equal(out[line_idx, 4:6], coords[line_idx, 4:6])


def test_random_coord_jitter_zero_std_is_identity() -> None:
    types, coords = _simple_outline()
    _, out = random_coord_jitter(types, coords, std=0.0)
    assert torch.equal(out, coords)


def test_random_coord_jitter_preserves_padding_zeros() -> None:
    types, coords = _simple_outline()
    g = torch.Generator().manual_seed(1)
    _, out = random_coord_jitter(types, coords, std=1.0, generator=g)
    assert _close_end_zeros(types, out)


def test_random_coord_jitter_quad_pair1_stays_zero() -> None:
    types, coords = _quad_outline()
    g = torch.Generator().manual_seed(2)
    _, out = random_coord_jitter(types, coords, std=1.0, generator=g)

    quad_idx = types.tolist().index(CommandType.QUAD_TO.value)
    assert out[quad_idx, 2].item() == pytest.approx(0.0)
    assert out[quad_idx, 3].item() == pytest.approx(0.0)


def test_random_coord_jitter_jitters_all_cubic_pairs() -> None:
    types, coords = _cubic_outline()
    g = torch.Generator().manual_seed(7)
    _, out = random_coord_jitter(types, coords, std=0.5, generator=g)

    curve_idx = types.tolist().index(CommandType.CURVE_TO.value)
    assert not torch.equal(out[curve_idx, 0:2], coords[curve_idx, 0:2])
    assert not torch.equal(out[curve_idx, 2:4], coords[curve_idx, 2:4])
    assert not torch.equal(out[curve_idx, 4:6], coords[curve_idx, 4:6])


def test_random_coord_jitter_does_not_modify_types() -> None:
    types, coords = _simple_outline()
    g = torch.Generator().manual_seed(0)
    out_types, _ = random_coord_jitter(types, coords, std=0.1, generator=g)
    assert torch.equal(out_types, types)


def test_random_coord_jitter_negative_std_raises() -> None:
    types, coords = _simple_outline()
    with pytest.raises(ValueError, match="std must be non-negative"):
        random_coord_jitter(types, coords, std=-0.1)


def test_random_coord_jitter_deterministic_with_generator() -> None:
    types, coords = _simple_outline()
    g1 = torch.Generator().manual_seed(99)
    g2 = torch.Generator().manual_seed(99)
    _, out1 = random_coord_jitter(types, coords, std=0.05, generator=g1)
    _, out2 = random_coord_jitter(types, coords, std=0.05, generator=g2)
    assert torch.equal(out1, out2)
