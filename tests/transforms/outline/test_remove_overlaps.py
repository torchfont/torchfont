import pytest
import torch

from tests._pairs import remove_overlaps
from torchfont import ElementType, Outline
from torchfont.transforms import RemoveOverlaps


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


def test_remove_overlaps_verify_keeps_a_correct_merge() -> None:
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
            [0, 0, 0, 0, 100, 100],
            [0, 0, 0, 0, 600, 100],
            [0, 0, 0, 0, 600, 600],
            [0, 0, 0, 0, 100, 600],
            [0, 0, 0, 0, 100, 100],
            [0, 0, 0, 0, 300, 100],
            [0, 0, 0, 0, 800, 100],
            [0, 0, 0, 0, 800, 600],
            [0, 0, 0, 0, 300, 600],
            [0, 0, 0, 0, 100, 100],
            [0, 0, 0, 0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )

    unverified = RemoveOverlaps(verify=False)(Outline(types, coords))
    verified = RemoveOverlaps(verify=True)(Outline(types, coords))

    assert torch.equal(unverified.types, verified.types)
    assert torch.allclose(unverified.coords, verified.coords)
    assert verified.types.tolist().count(ElementType.MOVE_TO.value) == 1


@pytest.mark.parametrize("verify_size", [0, 4097])
def test_remove_overlaps_rejects_invalid_verify_size(verify_size: int) -> None:
    types = torch.tensor([ElementType.END.value])
    coords = torch.zeros((1, 6))

    with pytest.raises(ValueError, match="verify_size must be between 1 and 4096"):
        RemoveOverlaps(verify=True, verify_size=verify_size)(Outline(types, coords))
