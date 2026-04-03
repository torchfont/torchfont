import pytest
import torch

from torchfont.datasets import GlyphSample
from torchfont.io import CommandType
from torchfont.transforms import Compose, LimitSequenceLength, Patchify, QuadToCubic


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
        metrics=torch.zeros(6),
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

    sample = GlyphSample(
        types=types, coords=coords, metrics=torch.zeros(6), style_idx=1, content_idx=2
    )
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

    sample = GlyphSample(
        types=types, coords=coords, metrics=torch.zeros(6), style_idx=4, content_idx=5
    )
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
        metrics=torch.zeros(6),
        style_idx=11,
        content_idx=13,
    )

    out = transform(sample)

    assert out.style_idx == sample.style_idx
    assert out.content_idx == sample.content_idx
    assert out.types.shape[0] == 2
    assert out.coords.shape[0] == 2


@pytest.mark.parametrize(
    ("transform_cls", "kwargs", "expected_message"),
    [
        (LimitSequenceLength, {"max_len": -1}, "max_len must be >= 0"),
        (Patchify, {"patch_size": 0}, "patch_size must be >= 1"),
        (Patchify, {"patch_size": -1}, "patch_size must be >= 1"),
    ],
)
def test_transform_constructors_validate_invalid_arguments(
    transform_cls: type[LimitSequenceLength] | type[Patchify],
    kwargs: dict[str, int],
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        transform_cls(**kwargs)


@pytest.mark.parametrize(
    ("sample", "expected_message"),
    [
        (
            GlyphSample(
                types=torch.tensor(1, dtype=torch.long),
                coords=torch.zeros(1, 6),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            r"LimitSequenceLength expects types\.ndim == 1; got 0",
        ),
        (
            GlyphSample(
                types=torch.tensor([1, 2], dtype=torch.long),
                coords=torch.zeros(2),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            r"LimitSequenceLength expects coords\.ndim == 2; got 1",
        ),
        (
            GlyphSample(
                types=torch.tensor([1, 2], dtype=torch.long),
                coords=torch.zeros(2, 5),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            r"LimitSequenceLength expects coords\.shape\[1\] == 6; got 5",
        ),
        (
            GlyphSample(
                types=torch.tensor([1, 2], dtype=torch.long),
                coords=torch.zeros(3, 6),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            (
                r"LimitSequenceLength expects types\.shape\[0\] and "
                r"coords\.shape\[0\] to match; got 2 and 3"
            ),
        ),
    ],
)
def test_limit_sequence_length_rejects_malformed_samples(
    sample: GlyphSample,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        LimitSequenceLength(2)(sample)


@pytest.mark.parametrize(
    ("sample", "expected_message"),
    [
        (
            GlyphSample(
                types=torch.tensor(1, dtype=torch.long),
                coords=torch.zeros(1, 6),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            r"Patchify expects types\.ndim == 1; got 0",
        ),
        (
            GlyphSample(
                types=torch.tensor([1, 2], dtype=torch.long),
                coords=torch.zeros(2),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            r"Patchify expects coords\.ndim == 2; got 1",
        ),
        (
            GlyphSample(
                types=torch.tensor([1, 2], dtype=torch.long),
                coords=torch.zeros(2, 5),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            r"Patchify expects coords\.shape\[1\] == 6; got 5",
        ),
        (
            GlyphSample(
                types=torch.tensor([1, 2], dtype=torch.long),
                coords=torch.zeros(3, 6),
                metrics=torch.zeros(6),
                style_idx=0,
                content_idx=0,
            ),
            (
                r"Patchify expects types\.shape\[0\] and coords\.shape\[0\] "
                r"to match; got 2 and 3"
            ),
        ),
    ],
)
def test_patchify_rejects_malformed_samples(
    sample: GlyphSample,
    expected_message: str,
) -> None:
    with pytest.raises(ValueError, match=expected_message):
        Patchify(2)(sample)
