import pytest
import torch

from torchfont.io import ElementType
from torchfont.transforms import randomize_subpath_order, render_bitmap


@pytest.fixture
def two_squares() -> tuple[torch.Tensor, torch.Tensor]:
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
    )
    coords = torch.tensor(
        [
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.4, 0.0],
            [0, 0, 0, 0, 0.4, 0.4],
            [0, 0, 0, 0, 0.0, 0.4],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.6, 0.6],
            [0, 0, 0, 0, 1.0, 0.6],
            [0, 0, 0, 0, 1.0, 1.0],
            [0, 0, 0, 0, 0.6, 1.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return types, coords


def test_randomize_subpath_order_is_reproducible(
    two_squares: tuple[torch.Tensor, torch.Tensor],
) -> None:
    first = torch.Generator().manual_seed(4)
    second = torch.Generator().manual_seed(4)
    output1 = randomize_subpath_order(*two_squares, generator=first)
    output2 = randomize_subpath_order(*two_squares, generator=second)
    assert torch.equal(output1[0], output2[0])
    assert torch.equal(output1[1], output2[1])


def test_randomize_subpath_order_preserves_rendering(
    two_squares: tuple[torch.Tensor, torch.Tensor],
) -> None:
    output = randomize_subpath_order(
        *two_squares,
        generator=torch.Generator().manual_seed(4),
    )
    before = render_bitmap(*two_squares, size=64)
    after = render_bitmap(*output, size=64)
    assert torch.equal(after, before)


def test_randomize_subpath_order_preserves_each_subpath(
    two_squares: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = two_squares
    out_types, out_coords = randomize_subpath_order(
        types,
        coords,
        generator=torch.Generator().manual_seed(4),
    )
    input_blocks = {
        (tuple(types[:5].tolist()), tuple(coords[:5].flatten().tolist())),
        (tuple(types[5:10].tolist()), tuple(coords[5:10].flatten().tolist())),
    }
    output_blocks = {
        (tuple(out_types[:5].tolist()), tuple(out_coords[:5].flatten().tolist())),
        (tuple(out_types[5:10].tolist()), tuple(out_coords[5:10].flatten().tolist())),
    }
    assert output_blocks == input_blocks
    assert out_coords[0, 4:6].tolist() == pytest.approx([0.6, 0.6])
    assert out_types[-1].item() == ElementType.END.value
