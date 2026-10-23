from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

from torchfont import ElementType, Outline


@pytest.fixture
def triangle() -> Outline:
    types = torch.tensor(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.LINE_TO,
            ElementType.CLOSE,
            ElementType.END,
        ],
        dtype=torch.long,
    )
    coords = torch.zeros(5, 6)
    coords[1, 4:] = torch.tensor([1.0, 0.0])
    coords[2, 4:] = torch.tensor([0.5, 1.0])
    return Outline(types, coords)


def test_outline_exposes_tensor_metadata(triangle: Outline) -> None:
    assert triangle.shape == (5,)
    assert triangle.num_elements == 5
    assert triangle.dtype is torch.float32
    assert triangle.device == torch.device("cpu")


def test_to_casts_only_coords(triangle: Outline) -> None:
    moved = triangle.to(torch.float64)

    assert moved.dtype is torch.float64
    assert moved.types.dtype is torch.long
    assert torch.equal(moved.types, triangle.types)


def test_to_accepts_device_and_dtype_together(triangle: Outline) -> None:
    moved = triangle.to("cpu", torch.float64)

    assert moved.device == torch.device("cpu")
    assert moved.dtype is torch.float64


def test_to_rejects_a_non_floating_dtype(triangle: Outline) -> None:
    with pytest.raises(TypeError, match="dtype must be floating point"):
        triangle.to(torch.int32)


def test_to_rejects_a_duplicated_dtype(triangle: Outline) -> None:
    with pytest.raises(TypeError, match="both positionally and by keyword"):
        triangle.to(torch.float64, torch.float64)


def test_outline_is_a_pytree_leaf(triangle: Outline) -> None:
    leaves, _ = pytree.tree_flatten(triangle)

    assert leaves == [triangle]


def test_indexing_keeps_the_coordinate_axis(triangle: Outline) -> None:
    head = triangle[:2]

    assert head.shape == (2,)
    assert head.coords.shape == (2, 6)


def test_indexing_rejects_removing_the_element_dimension(triangle: Outline) -> None:
    with pytest.raises(IndexError, match="preserve the outline element dimension"):
        triangle[0]


def test_outline_aliases_its_tensors(triangle: Outline) -> None:
    alias = Outline(triangle.types, triangle.coords)
    triangle.coords[0, 4] = 5.0

    assert alias.types is triangle.types
    assert alias.coords is triangle.coords
    assert alias.coords[0, 4] == 5.0
    with pytest.raises(AttributeError):
        triangle.types = triangle.types  # ty: ignore[invalid-assignment]


def test_repr_summarises_instead_of_dumping_tensors(triangle: Outline) -> None:
    assert repr(triangle) == ("Outline(shape=(5,), dtype=torch.float32, device=cpu)")
