import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import (
    HorizontalFlip,
    NormalizeWinding,
    RandomHorizontalFlip,
    RandomRemoveOverlaps,
    RandomVerticalFlip,
    RandomWinding,
    RemoveOverlaps,
    VerticalFlip,
    functional,
)
from torchfont.transforms._transform import Transform


@pytest.mark.parametrize("return_to_start", [False, True])
@pytest.mark.parametrize(
    ("transform", "axis", "reconstructs"),
    [
        (NormalizeWinding(), None, False),
        (RandomWinding(), None, False),
        (RemoveOverlaps(), None, True),
        (RandomRemoveOverlaps(), None, True),
        (HorizontalFlip(), -1, False),
        (VerticalFlip(), -2, False),
        (RandomHorizontalFlip(p=1), -1, False),
        (RandomVerticalFlip(p=1), -2, False),
    ],
)
def test_mixed_open_and_closed_subpaths_preserve_fill(
    transform: Transform, axis: int | None, *, reconstructs: bool, return_to_start: bool
) -> None:
    outer = [(0, 0), (2, 0), (2, 2), (0, 2)]
    if return_to_start:
        outer.append(outer[0])
    inner = [(0.5, 0.5), (0.5, 1.5), (1.5, 1.5), (1.5, 0.5)]
    types = torch.tensor(
        [ElementType.MOVE_TO]
        + [ElementType.LINE_TO] * (len(outer) - 1)
        + [ElementType.MOVE_TO]
        + [ElementType.LINE_TO] * 3
        + [ElementType.CLOSE, ElementType.END]
    )
    coords = torch.tensor(
        [[0, 0, 0, 0, x, y] for x, y in outer + inner] + [[0] * 6] * 2,
        dtype=torch.float32,
    )
    outline = Outline(types, coords)
    output = transform(outline)
    for fill_rule in ("winding", "even_odd"):
        original = functional.render_bitmap(
            outline, 64, fill_rule=fill_rule, antialias=False
        )
        expected = original if axis is None else original.flip(axis)
        assert torch.equal(
            functional.render_bitmap(output, 64, fill_rule=fill_rule, antialias=False),
            expected,
        )
    moves = int((output.types == ElementType.MOVE_TO).sum())
    closes = int((output.types == ElementType.CLOSE).sum())
    assert moves - closes == (0 if reconstructs else 1)
