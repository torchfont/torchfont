import torch

from torchfont.io import ElementType
from torchfont.transforms import remove_overlaps


def test_remove_overlaps_merges_overlapping_subpaths() -> None:
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
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 2.0, 0.0],
            [0, 0, 0, 0, 2.0, 2.0],
            [0, 0, 0, 0, 0.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 1.0, 0.0],
            [0, 0, 0, 0, 3.0, 0.0],
            [0, 0, 0, 0, 3.0, 2.0],
            [0, 0, 0, 0, 1.0, 2.0],
            [0, 0, 0, 0, 0.0, 0.0],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    out_types, out_coords = remove_overlaps(types, coords)

    assert out_types[-1].item() == ElementType.END.value
    assert out_types.tolist().count(ElementType.MOVE_TO.value) == 1
    assert out_types.tolist().count(ElementType.CLOSE.value) == 1
    expected = torch.tensor([0.0, 0.0, 3.0, 2.0])
    actual = torch.tensor(
        [
            out_coords[:, 4].min(),
            out_coords[:, 5].min(),
            out_coords[:, 4].max(),
            out_coords[:, 5].max(),
        ]
    )
    assert torch.allclose(actual, expected)
