import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import render_bitmap

from ._helpers import _occupied_size


def test_render_bitmap_supports_coordinate_mapping_modes() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.75, 0.25],
            [0.0, 0.0, 0.0, 0.0, 0.75, 0.50],
            [0.0, 0.0, 0.0, 0.0, 0.25, 0.50],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    fixed = _occupied_size(render_bitmap(types, coords, size=64, mode="fixed"))
    bbox = _occupied_size(render_bitmap(types, coords, size=64, mode="bbox"))
    bbox_square = _occupied_size(
        render_bitmap(types, coords, size=64, mode="bbox_square")
    )
    default = render_bitmap(types, coords, size=64)

    assert fixed == (22, 11)
    assert bbox == (22, 11)
    assert bbox_square == (64, 32)
    assert torch.equal(
        default, render_bitmap(types, coords, size=64, mode="bbox_square")
    )


def test_render_bitmap_bbox_returns_variable_size() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.10, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.60, 0.10],
            [0.0, 0.0, 0.0, 0.0, 0.60, 0.35],
            [0.0, 0.0, 0.0, 0.0, 0.10, 0.35],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    bitmap = render_bitmap(types, coords, size=64, mode="bbox")

    assert bitmap.shape == (11, 22)
    assert bitmap.device.type == "cpu"


def test_render_bitmap_rejects_unknown_mode() -> None:
    types = torch.tensor([ElementType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="mode must be one of"):
        render_bitmap(types, coords, mode="unknown")  # ty: ignore[invalid-argument-type]


def test_render_bitmap_bbox_empty_outline_returns_empty_bitmap() -> None:
    types = torch.tensor([ElementType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    bitmap = render_bitmap(types, coords, mode="bbox")

    assert bitmap.shape == (0, 0)


def test_render_bitmap_bbox_rejects_oversized_output() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 200.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 200.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    with pytest.raises(ValueError, match="bbox output dimensions"):
        render_bitmap(types, coords, mode="bbox")
