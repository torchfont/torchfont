import pytest
import torch

from torchfont.datasets import GlyphSample
from torchfont.io import CommandType
from torchfont.transforms import patchify, quad_to_cubic, render_bitmap

_ZERO_METRICS = torch.zeros(15, dtype=torch.float32)  # placeholder for transform tests


def test_quad_to_cubic_converts_quadratic_segments() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.QUAD_TO.value,
            CommandType.LINE_TO.value,
            CommandType.QUAD_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # moveTo(0, 0)
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],  # quadTo((0, 1), (1, 1))
            [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],  # lineTo(2, 1)
            [3.0, 2.0, 0.0, 0.0, 4.0, 1.0],  # quadTo((3, 2), (4, 1))
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # close
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],  # end
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = quad_to_cubic(types, coords)

    expected_types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.CURVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CURVE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
        ],
        dtype=torch.long,
    )
    expected_coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            [8.0 / 3.0, 5.0 / 3.0, 10.0 / 3.0, 5.0 / 3.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    assert torch.equal(out_types, expected_types)
    assert torch.allclose(out_coords, expected_coords)


def test_quad_to_cubic_returns_inputs_when_no_quadratic_segments() -> None:
    types = torch.tensor(
        [CommandType.MOVE_TO.value, CommandType.LINE_TO.value, CommandType.END.value],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = quad_to_cubic(types, coords)

    assert out_types is types
    assert out_coords is coords


def test_quad_to_cubic_supports_extra_leading_dimensions() -> None:
    types = torch.tensor(
        [
            [CommandType.MOVE_TO.value, CommandType.QUAD_TO.value],
            [CommandType.LINE_TO.value, CommandType.END.value],
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [[0.0, 0.0, 0.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0, 1.0, 1.0]],
            [[0.0, 0.0, 0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]],
        ],
        dtype=torch.float32,
    )

    sample = GlyphSample(
        types=types,
        coords=coords,
        style_idx=4,
        content_idx=5,
        metrics=_ZERO_METRICS,
        glyph_name="",
    )
    out_types, out_coords = quad_to_cubic(sample.types, sample.coords)

    assert out_types.shape == types.shape
    assert out_coords.shape == coords.shape
    assert out_types[0, 1].item() == CommandType.CURVE_TO.value
    assert torch.allclose(
        out_coords[0, 1],
        torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )


def test_quad_to_cubic_keeps_batched_sequences_independent() -> None:
    types = torch.tensor(
        [
            [
                CommandType.MOVE_TO.value,
                CommandType.LINE_TO.value,
                CommandType.END.value,
            ],
            [
                CommandType.QUAD_TO.value,
                CommandType.LINE_TO.value,
                CommandType.END.value,
            ],
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [
                [0.0, 0.0, 0.0, 0.0, 7.0, 7.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
                [0.0, 0.0, 0.0, 0.0, 9.0, 9.0],
            ],
            [
                [0.0, 3.0, 0.0, 0.0, 3.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 3.0],
                [0.0, 0.0, 0.0, 0.0, 4.0, 3.0],
            ],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = quad_to_cubic(types, coords)

    assert out_types[1, 0].item() == CommandType.CURVE_TO.value
    assert torch.allclose(
        out_coords[1, 0],
        torch.tensor([0.0, 2.0, 1.0, 3.0, 3.0, 3.0], dtype=torch.float32),
    )


def test_patchify_reshapes_exact_multiple() -> None:
    types = torch.tensor([1, 2, 3, 4], dtype=torch.long)
    coords = torch.zeros(4, 6, dtype=torch.float32)

    out_types, out_coords = patchify(types, coords, patch_size=2)

    assert out_types.shape == (2, 2)
    assert out_coords.shape == (2, 2, 6)
    assert torch.equal(out_types, types.view(2, 2))


def test_patchify_pads_when_not_exact_multiple() -> None:
    types = torch.tensor([1, 2, 3], dtype=torch.long)
    coords = torch.zeros(3, 6, dtype=torch.float32)

    out_types, out_coords = patchify(types, coords, patch_size=2)

    assert out_types.shape == (2, 2)
    assert out_coords.shape == (2, 2, 6)
    assert out_types[1, 1].item() == 0


def test_patchify_raises_for_invalid_patch_size() -> None:
    types = torch.tensor([1], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="patch_size must be >= 1"):
        patchify(types, coords, patch_size=0)


def test_render_bitmap_supports_coordinate_mapping_modes() -> None:
    types = torch.tensor(
        [
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
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
            CommandType.MOVE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.LINE_TO.value,
            CommandType.CLOSE.value,
            CommandType.END.value,
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


def test_render_bitmap_rejects_unknown_mode() -> None:
    types = torch.tensor([CommandType.END.value], dtype=torch.long)
    coords = torch.zeros(1, 6, dtype=torch.float32)

    with pytest.raises(ValueError, match="mode must be one of"):
        render_bitmap(types, coords, mode="unknown")  # type: ignore[arg-type]


def _occupied_size(bitmap: torch.Tensor) -> tuple[int, int]:
    ys, xs = torch.nonzero(bitmap > 0, as_tuple=True)
    width = int(xs.max().item()) - int(xs.min().item()) + 1
    height = int(ys.max().item()) - int(ys.min().item()) + 1
    return width, height
