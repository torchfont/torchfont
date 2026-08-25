import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import Affine, Compose, SplitSubpaths, functional


def _outline(types: list[ElementType]) -> Outline:
    type_tensor = torch.tensor(types, dtype=torch.long)
    coords = torch.arange(len(types) * 6, dtype=torch.float32).view(-1, 6)
    return Outline(type_tensor, coords)


def test_split_subpaths_returns_independent_encodings() -> None:
    outline = _outline(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.CLOSE,
            ElementType.MOVE_TO,
            ElementType.QUAD_TO,
            ElementType.END,
        ]
    )

    first, second = functional.split_subpaths(outline)

    assert first.types.tolist() == [
        ElementType.MOVE_TO,
        ElementType.LINE_TO,
        ElementType.CLOSE,
        ElementType.END,
    ]
    assert second.types.tolist() == [
        ElementType.MOVE_TO,
        ElementType.QUAD_TO,
        ElementType.END,
    ]
    assert torch.equal(first.coords[:2, 4:6], outline.coords[:2, 4:6])
    assert torch.equal(second.coords[0, 4:6], outline.coords[3, 4:6])
    assert torch.equal(second.coords[1, (0, 1, 4, 5)], outline.coords[4, (0, 1, 4, 5)])
    assert torch.count_nonzero(first.coords[2]) == 0
    assert torch.count_nonzero(first.coords[-1]) == 0


def test_split_subpaths_rejects_coordinate_gradients() -> None:
    outline = _outline([ElementType.MOVE_TO, ElementType.LINE_TO, ElementType.END])
    outline.coords.requires_grad_()

    with pytest.raises(RuntimeError, match=r"split_subpaths.*not differentiable"):
        functional.split_subpaths(outline)


def test_split_subpaths_returns_empty_tuple_without_subpaths() -> None:
    outline = _outline([ElementType.END])

    assert functional.split_subpaths(outline) == ()


def test_split_subpaths_rejects_an_element_outside_subpath() -> None:
    outline = _outline([ElementType.LINE_TO, ElementType.END])

    with pytest.raises(ValueError, match="requires a preceding MOVE_TO"):
        functional.split_subpaths(outline)


def test_split_subpaths_transform_preserves_enclosing_data() -> None:
    outline = _outline(
        [
            ElementType.MOVE_TO,
            ElementType.CLOSE,
            ElementType.MOVE_TO,
            ElementType.CLOSE,
            ElementType.END,
        ]
    )

    result = SplitSubpaths()({"outline": outline, "label": 4})

    assert result["label"] == 4
    assert isinstance(result["outline"], tuple)
    assert len(result["outline"]) == 2


def test_following_transform_processes_each_subpath() -> None:
    outline = _outline(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.END,
        ]
    )

    subpaths = Compose([SplitSubpaths(), Affine(angle=10.0)])(outline)

    assert len(subpaths) == 2
    assert all(isinstance(subpath, Outline) for subpath in subpaths)
    assert all(
        not torch.equal(subpath.coords[:-1], outline.coords[:2]) for subpath in subpaths
    )
