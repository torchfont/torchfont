import torch

from torchfont.datasets import GlyphSample
from torchfont.io import CommandType
from torchfont.transforms import Compose, LimitSequenceLength, QuadToCubic


def test_quad_to_cubic_converts_quadratic_segments() -> None:
    transform = QuadToCubic()

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

    sample = GlyphSample(
        types=types,
        coords=coords,
        style_idx=3,
        content_idx=7,
    )
    out = transform(sample)

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

    assert torch.equal(out.types, expected_types)
    assert torch.allclose(out.coords, expected_coords)
    assert out.style_idx == sample.style_idx
    assert out.content_idx == sample.content_idx


def test_quad_to_cubic_returns_inputs_when_no_quadratic_segments() -> None:
    transform = QuadToCubic()

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

    sample = GlyphSample(types=types, coords=coords, style_idx=1, content_idx=2)
    out = transform(sample)

    assert out is sample


def test_quad_to_cubic_supports_patchified_shapes() -> None:
    transform = QuadToCubic()

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

    sample = GlyphSample(types=types, coords=coords, style_idx=4, content_idx=5)
    out = transform(sample)

    assert out.types.shape == types.shape
    assert out.coords.shape == coords.shape
    assert out.types[0, 1].item() == CommandType.CURVE_TO.value
    assert torch.allclose(
        out.coords[0, 1],
        torch.tensor([0.0, 2.0 / 3.0, 1.0 / 3.0, 1.0, 1.0, 1.0], dtype=torch.float32),
    )
    assert out.style_idx == sample.style_idx
    assert out.content_idx == sample.content_idx


def test_compose_preserves_metadata_across_sample_first_pipeline() -> None:
    transform = Compose((QuadToCubic(), LimitSequenceLength(max_len=2)))

    sample = GlyphSample(
        types=torch.tensor(
            [
                CommandType.MOVE_TO.value,
                CommandType.QUAD_TO.value,
                CommandType.END.value,
            ],
            dtype=torch.long,
        ),
        coords=torch.tensor(
            [
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
                [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
        style_idx=11,
        content_idx=13,
    )

    out = transform(sample)

    assert out.style_idx == sample.style_idx
    assert out.content_idx == sample.content_idx
    assert out.types.shape[0] == 2
    assert out.coords.shape[0] == 2
