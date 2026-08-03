import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import RenderBitmap
from torchfont.transforms.functional import render_bitmap

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

    outline = Outline(types, coords)
    fixed = _occupied_size(render_bitmap(outline, size=64, mode="fixed"))
    bbox = _occupied_size(render_bitmap(outline, size=64, mode="bbox"))
    bbox_square = _occupied_size(render_bitmap(outline, size=64, mode="bbox_square"))
    default = render_bitmap(outline, size=64)

    assert fixed == (22, 11)
    assert bbox == (22, 11)
    assert bbox_square == (64, 32)
    assert torch.equal(default, render_bitmap(outline, size=64, mode="bbox_square"))


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

    bitmap = render_bitmap(Outline(types, coords), size=64, mode="bbox")

    assert bitmap.shape == (11, 22)
    assert bitmap.device.type == "cpu"


def test_render_bitmap_supports_fill_rules() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
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
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.1],
            [0.0, 0.0, 0.0, 0.0, 0.9, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.1, 0.9],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.7, 0.3],
            [0.0, 0.0, 0.0, 0.0, 0.7, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    outline = Outline(types, coords)
    winding = render_bitmap(outline, size=64, fill_rule="winding")
    even_odd = render_bitmap(outline, size=64, fill_rule="even_odd")

    assert winding[32, 32] == 255
    assert even_odd[32, 32] == 0


def test_render_bitmap_can_disable_antialiasing() -> None:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.13, 0.17],
            [0.0, 0.0, 0.0, 0.0, 0.91, 0.23],
            [0.0, 0.0, 0.0, 0.0, 0.37, 0.89],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
            [0.0, 0.0, 0.0, 0.0, 0.00, 0.00],
        ],
        dtype=torch.float32,
    )

    outline = Outline(types, coords)
    antialiased = render_bitmap(outline)
    binary = render_bitmap(outline, antialias=False)
    transformed = RenderBitmap(antialias=False)(outline)

    assert torch.any((antialiased > 0) & (antialiased < 255))
    assert set(binary.unique().tolist()) <= {0, 255}
    assert not torch.equal(antialiased, binary)
    assert torch.equal(transformed, binary)


def test_render_bitmap_rejects_unknown_mode() -> None:
    types = torch.tensor([ElementType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="mode must be one of"):
        render_bitmap(
            Outline(types, coords),
            mode="unknown",  # ty: ignore[invalid-argument-type]
        )


def test_render_bitmap_bbox_empty_outline_returns_empty_bitmap() -> None:
    types = torch.tensor([ElementType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    bitmap = render_bitmap(Outline(types, coords), mode="bbox")

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
        render_bitmap(Outline(types, coords), mode="bbox")
