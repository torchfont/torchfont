import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import RandomizeSubpathOrder
from torchfont.transforms import functional as _functional


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
    outline = Outline(*two_squares)
    torch.manual_seed(4)
    output1 = RandomizeSubpathOrder()(outline)
    torch.manual_seed(4)
    output2 = RandomizeSubpathOrder()(outline)
    assert torch.equal(output1.types, output2.types)
    assert torch.equal(output1.coords, output2.coords)


def test_randomize_subpath_order_preserves_rendering(
    two_squares: tuple[torch.Tensor, torch.Tensor],
) -> None:
    outline = Outline(*two_squares)
    torch.manual_seed(4)
    output = RandomizeSubpathOrder()(outline)
    before = _functional.render_bitmap(outline, size=64)
    after = _functional.render_bitmap(output, size=64)
    assert torch.equal(after, before)


def test_randomize_subpath_order_preserves_each_subpath(
    two_squares: tuple[torch.Tensor, torch.Tensor],
) -> None:
    types, coords = two_squares
    torch.manual_seed(4)
    output = RandomizeSubpathOrder()(Outline(types, coords))
    out_types, out_coords = output.types, output.coords
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
