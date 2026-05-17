import torch

from torchfont.datasets import GlyphSample
from torchfont.io import CommandType
from torchfont.transforms import cubic_to_quad, quad_to_cubic

from ._helpers import (
    _CUBIC_CURVES,
    _ZERO_METRICS,
    _assert_single_cubic_matches,
    _cubic_segs_to_tensors,
    _line_path_to_tensors,
)


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
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 2.0, 1.0],
            [3.0, 2.0, 0.0, 0.0, 4.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
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
    assert out_types.device.type == "cpu"
    assert out_coords.device.type == "cpu"


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


def test_quad_to_cubic_can_merge_curves_in_same_transform() -> None:
    curve = _CUBIC_CURVES[0]
    types, coords = _cubic_segs_to_tensors([curve])
    q_types, q_coords = cubic_to_quad(types, coords)

    out_types, out_coords = quad_to_cubic(q_types, q_coords, merge_curves=True)

    _assert_single_cubic_matches(out_types, out_coords, curve)


def test_quad_to_cubic_merge_curves_runs_even_without_quadratics() -> None:
    types, coords = _line_path_to_tensors(
        [(0.0, 0.0), (1.0, 0.0), (2.0, 0.0), (3.0, 0.0)]
    )

    out_types, _out_coords = quad_to_cubic(types, coords, merge_curves=True)

    assert out_types.tolist().count(CommandType.LINE_TO.value) == 1
