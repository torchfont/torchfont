"""Rust kernels as custom operators: operator contracts and graph capture."""

from __future__ import annotations

import sys
from typing import TYPE_CHECKING

import pytest
import torch
from torch.library import CustomOpDef, opcheck

import torchfont._ops as ops
from torchfont import ElementType, Outline
from torchfont.transforms import functional as F  # noqa: N812

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator


@pytest.fixture(autouse=True)
def _capture_dynamic_output_shapes() -> Iterator[None]:
    """Enable data-dependent custom-op outputs on every supported PyTorch."""
    with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):  # noqa: SLF001
        yield


def _outline(elements: int = 5) -> Outline:
    types = torch.tensor(
        [ElementType.MOVE_TO]
        + [ElementType.CURVE_TO] * (elements - 3)
        + [ElementType.CLOSE, ElementType.END],
        dtype=torch.long,
    )
    generator = torch.Generator().manual_seed(elements)
    return Outline(types, torch.rand(elements, 6, generator=generator) * 0.8)


def _pipeline(types: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    outline = Outline(types, coords)
    outline = F.remove_overlaps(outline)
    outline = F.cubic_to_quad(outline)
    outline = F.merge_curves(outline)
    outline = F.affine(outline, angle=10.0, scale=1.1)
    outline = F.normalize_subpath_start_points(outline)
    return F.render_bitmap(outline, 32)


def _compile_pipeline(
    *, dynamic: bool | None = None
) -> Callable[[torch.Tensor, torch.Tensor], torch.Tensor]:
    backend = "eager" if sys.platform == "win32" else "inductor"
    return torch.compile(_pipeline, fullgraph=True, dynamic=dynamic, backend=backend)


def _cases() -> list[tuple[str, CustomOpDef, tuple[object, ...]]]:
    outline = _outline()
    pair = (outline.types, outline.coords)
    values = torch.rand(16, generator=torch.Generator().manual_seed(0))
    return [
        ("remove_overlaps", ops.remove_overlaps, pair),
        ("cubic_to_quad", ops.cubic_to_quad, pair),
        ("merge_curves", ops.merge_curves, pair),
        ("quad_to_cubic", ops.quad_to_cubic, (*pair, False)),
        ("quad_to_cubic_merged", ops.quad_to_cubic, (*pair, True)),
        (
            "normalize_subpath_start_points",
            ops.normalize_subpath_start_points,
            pair,
        ),
        ("reverse_closed_subpaths", ops.reverse_closed_subpaths, pair),
        ("bbox_center", ops.bbox_center, pair),
        ("set_subpath_start_points", ops.set_subpath_start_points, (*pair, values)),
        ("reorder_subpaths", ops.reorder_subpaths, (*pair, values)),
        ("remove_overlap_groups", ops.remove_overlap_groups, (*pair, values)),
        (
            "split_segments",
            ops.split_segments,
            (*pair, values, values, 0.5, [0.3, 0.7]),
        ),
        (
            "render_bitmap",
            ops.render_bitmap,
            (*pair, 32, "bbox_square", "winding", True),
        ),
        (
            "render_bitmap_bbox",
            ops.render_bitmap,
            (*pair, 32, "bbox", "winding", True),
        ),
    ]


@pytest.mark.parametrize(
    ("name", "op", "args"), _cases(), ids=[case[0] for case in _cases()]
)
def test_operator_passes_opcheck(
    name: str, op: CustomOpDef, args: tuple[object, ...]
) -> None:
    """Check schema, fake implementation, and autograd registration agree."""
    del name
    opcheck(op, args)


def test_pipeline_compiles_into_one_graph() -> None:
    explanation = torch._dynamo.explain(_pipeline)(  # noqa: SLF001
        _outline().types, _outline().coords
    )

    assert explanation.graph_break_count == 0
    assert explanation.graph_count == 1


def test_compiled_pipeline_matches_eager() -> None:
    torch._dynamo.reset()  # noqa: SLF001
    outline = _outline()
    compiled = _compile_pipeline()

    result = compiled(outline.types, outline.coords)

    assert torch.equal(result, _pipeline(outline.types, outline.coords))


@pytest.mark.parametrize("elements", [5, 9, 17, 33])
def test_compiled_pipeline_handles_varying_element_counts(elements: int) -> None:
    """Outline length is data-dependent, so the fakes must allow it to vary."""
    outline = _outline(elements)
    compiled = _compile_pipeline()

    result = compiled(outline.types, outline.coords)

    assert torch.equal(result, _pipeline(outline.types, outline.coords))


@pytest.mark.parametrize("dynamic", [None, True, False], ids=["auto", "on", "off"])
def test_pipeline_compiles_under_every_dynamic_setting(
    dynamic: bool | None,  # noqa: FBT001
) -> None:
    """``dynamic=True`` makes float parameters symbolic, so validation must trace."""
    torch._dynamo.reset()  # noqa: SLF001
    outline = _outline()
    compiled = _compile_pipeline(dynamic=dynamic)

    result = compiled(outline.types, outline.coords)

    assert torch.equal(result, _pipeline(outline.types, outline.coords))


def test_bbox_center_shifts_with_the_outline() -> None:
    outline = _outline()
    shift = (0.25, -0.5)

    center = ops.bbox_center(outline.types, outline.coords)
    moved = F.affine(outline, translate=shift)
    moved_center = ops.bbox_center(moved.types, moved.coords)

    assert center.shape == (2,)
    assert torch.allclose(moved_center, center + torch.tensor(shift), atol=1e-6)


def test_bbox_center_of_an_empty_outline_is_the_origin() -> None:
    empty = Outline(
        torch.tensor([ElementType.END], dtype=torch.long), torch.zeros(1, 6)
    )

    assert torch.equal(ops.bbox_center(empty.types, empty.coords), torch.zeros(2))


def test_native_operators_accept_float32() -> None:
    dtype = torch.float32
    outline = _outline().to(dtype)

    converted = F.quad_to_cubic(outline)
    transformed = F.affine(outline, angle=10.0)
    bitmap = F.render_bitmap(outline, 32)

    assert converted.dtype is dtype
    assert transformed.dtype is dtype
    assert bitmap.dtype is torch.uint8


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float64])
def test_native_operators_reject_unsupported_dtypes(dtype: torch.dtype) -> None:
    outline = _outline().to(dtype)

    with pytest.raises(TypeError, match=r"coords must have dtype torch\.float32"):
        F.quad_to_cubic(outline)
