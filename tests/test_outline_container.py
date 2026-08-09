"""Tensor-like container behaviour of :class:`torchfont.Outline`."""

from __future__ import annotations

import pytest
import torch
from torch.utils import _pytree as pytree

from torchfont import ElementType, Outline, pad_outlines, unpad_outlines
from torchfont.transforms import Affine


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
    assert triangle.batch_shape == ()
    assert triangle.num_elements == 5
    assert not triangle.is_batched
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


def test_tree_map_moves_an_outline(triangle: Outline) -> None:
    moved = pytree.tree_map(lambda leaf: leaf.to(torch.float64), triangle)

    assert moved.dtype is torch.float64


def test_registering_outline_as_a_pytree_node_would_break_transforms(
    triangle: Outline,
) -> None:
    """Guard the reason ``Outline`` must stay a leaf.

    ``Transform`` selects work with ``isinstance(leaf, Outline)`` after
    ``tree_flatten``. If ``Outline`` decomposed into bare tensors, nothing would
    match and every transform would silently return its input.
    """
    flipped = Affine(angle=10.0)(triangle)

    assert not torch.equal(flipped.coords, triangle.coords)
    assert pytree.tree_flatten(triangle)[0] == [triangle]


def test_padding_mask_marks_padding(triangle: Outline) -> None:
    batch = pad_outlines([triangle, triangle[:3]])

    assert batch.padding_mask.tolist() == [
        [False] * 5,
        [False, False, False, True, True],
    ]


def test_indexing_keeps_the_coordinate_axis(triangle: Outline) -> None:
    head = triangle[:2]

    assert head.shape == (2,)
    assert head.coords.shape == (2, 6)


def test_indexing_with_ellipsis_keeps_the_coordinate_axis() -> None:
    outline = Outline(
        torch.ones(2, 3, 4, dtype=torch.long),
        torch.zeros(2, 3, 4, 6),
    )

    result = outline[..., :2]

    assert result.shape == (2, 3, 2)
    assert result.coords.shape == (2, 3, 2, 6)


def test_indexing_rejects_removing_the_element_dimension(triangle: Outline) -> None:
    with pytest.raises(IndexError, match="preserve the outline element dimension"):
        triangle[0]


def test_pad_outlines_stacks_to_the_longest(triangle: Outline) -> None:
    batch = pad_outlines([triangle, triangle[:2], triangle[:4]])

    assert batch.shape == (3, 5)
    assert batch.coords.shape == (3, 5, 6)
    assert batch.is_batched
    assert len(batch) == 3


def test_pad_outlines_then_unpad_round_trips(triangle: Outline) -> None:
    parts = [triangle, triangle[:2], triangle[:4]]
    restored = unpad_outlines(pad_outlines(parts))

    assert len(restored) == len(parts)
    for actual, expected in zip(restored, parts, strict=True):
        assert torch.equal(actual.types, expected.types)
        assert torch.equal(actual.coords, expected.coords)


def test_unbind_preserves_padding(triangle: Outline) -> None:
    batch = pad_outlines([triangle, triangle[:2]])

    assert [part.shape for part in batch.unbind()] == [(5,), (5,)]


def test_unbind_multidimensional_batch_keeps_padding() -> None:
    outline = Outline(
        torch.ones(2, 3, 4, dtype=torch.long),
        torch.zeros(2, 3, 4, 6),
    )

    parts = outline.unbind()

    assert [part.shape for part in parts] == [(3, 4), (3, 4)]


def test_unbind_rejects_a_single_outline(triangle: Outline) -> None:
    with pytest.raises(ValueError, match="unbind requires a batched outline"):
        triangle.unbind()


def test_unpad_outlines_rejects_a_single_outline(triangle: Outline) -> None:
    with pytest.raises(ValueError, match="requires exactly one batch dimension"):
        unpad_outlines(triangle)


def test_unpad_outlines_rejects_a_multidimensional_batch() -> None:
    outline = Outline(
        torch.ones(2, 3, 4, dtype=torch.long),
        torch.zeros(2, 3, 4, 6),
    )

    with pytest.raises(ValueError, match="requires exactly one batch dimension"):
        unpad_outlines(outline)


def test_pad_outlines_rejects_an_empty_sequence() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        pad_outlines([])


def test_pad_outlines_rejects_a_batched_input(triangle: Outline) -> None:
    """The copy into the padded tensor raises on a shape mismatch."""
    batch = pad_outlines([triangle, triangle])

    with pytest.raises(RuntimeError):
        pad_outlines([batch])


def test_pad_outlines_rejects_mixed_dtypes(triangle: Outline) -> None:
    """Unlike a shape or device mismatch, a dtype mismatch would cast silently."""
    with pytest.raises(ValueError, match="share one coords dtype"):
        pad_outlines([triangle, triangle.to(torch.float64)])


def test_outline_aliases_its_tensors(triangle: Outline) -> None:
    """``frozen=True`` blocks rebinding attributes, not mutating the tensors."""
    triangle.coords[0, 4] = 5.0

    assert triangle.coords[0, 4] == 5.0
    with pytest.raises(AttributeError):
        triangle.types = triangle.types  # ty: ignore[invalid-assignment]


def test_repr_summarises_instead_of_dumping_tensors(triangle: Outline) -> None:
    assert repr(triangle) == ("Outline(shape=(5,), dtype=torch.float32, device=cpu)")
