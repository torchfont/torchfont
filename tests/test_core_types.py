import pytest
import torch
from torch.utils import _pytree as pytree

import torchfont
from torchfont import (
    COORD_DIM,
    TYPE_DIM,
    ElementType,
    FontRef,
    GlyphData,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
    GlyphSample,
    Outline,
)


def test_core_types_are_exported_from_package_root() -> None:
    assert torchfont.COORD_DIM == COORD_DIM
    assert torchfont.TYPE_DIM == TYPE_DIM
    assert torchfont.ElementType is ElementType
    assert torchfont.FontRef is FontRef
    assert torchfont.GlyphData is GlyphData
    assert torchfont.GlyphIdData is GlyphIdData
    assert torchfont.GlyphIdSample is GlyphIdSample
    assert torchfont.GlyphRef is GlyphRef
    assert torchfont.GlyphSample is GlyphSample
    assert torchfont.Outline is Outline


def test_glyph_ref_identifies_face_and_glyph() -> None:
    ref = GlyphRef(FontRef("font.ttf", 0), 36)

    assert ref.glyph_id == 36


@pytest.mark.parametrize(
    ("types", "coords", "match"),
    [
        (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 6)),
            "exactly one more dimension",
        ),
        (
            torch.zeros(1, dtype=torch.long),
            torch.zeros((1, 5)),
            "coords must have shape",
        ),
        (
            torch.zeros(1, dtype=torch.long),
            torch.zeros((2, 6)),
            "types shape must match coords",
        ),
        (
            torch.zeros((2, 3), dtype=torch.long),
            torch.zeros((2, 4, 6)),
            "types shape must match coords",
        ),
    ],
)
def test_outline_rejects_incompatible_shapes(
    types: torch.Tensor,
    coords: torch.Tensor,
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        Outline(types, coords)


def test_outline_rejects_invalid_dtypes() -> None:
    with pytest.raises(TypeError, match=r"types must have dtype torch\.long"):
        Outline(torch.zeros(1), torch.zeros((1, 6)))
    with pytest.raises(TypeError, match="coords must have a floating point dtype"):
        Outline(
            torch.zeros(1, dtype=torch.long), torch.zeros((1, 6), dtype=torch.int64)
        )


def test_outline_rejects_mismatched_devices() -> None:
    with pytest.raises(ValueError, match="must be on the same device"):
        Outline(
            torch.zeros(1, dtype=torch.long, device="meta"),
            torch.zeros((1, 6)),
        )


@pytest.mark.parametrize(
    "dtype", [torch.float16, torch.bfloat16, torch.float32, torch.float64]
)
def test_outline_accepts_any_floating_coords_dtype(dtype: torch.dtype) -> None:
    outline = Outline(
        torch.zeros(1, dtype=torch.long), torch.zeros((1, 6), dtype=dtype)
    )

    assert outline.dtype is dtype


def test_outline_accepts_batch_dimensions() -> None:
    outline = Outline(torch.zeros((4, 9), dtype=torch.long), torch.zeros((4, 9, 6)))

    assert outline.is_batched
    assert outline.shape == (4, 9)
    assert outline.batch_shape == (4,)
    assert outline.num_elements == 9
    assert len(outline) == 4


def test_outline_keeps_identity_equality() -> None:
    types = torch.tensor([1, 6], dtype=torch.long)
    coords = torch.zeros((2, 6))
    first = Outline(types, coords)
    second = Outline(types.clone(), coords.clone())
    assert first is not second
    assert first != second
    assert hash(first) == object.__hash__(first)


def test_glyph_data_keeps_identity_equality() -> None:
    types = torch.tensor([1, 6], dtype=torch.long)
    coords = torch.zeros((2, 6))
    sample = GlyphSample(GlyphRef(FontRef("font.ttf", 0), 36), 0x41, 0, 0)

    def build(outline: Outline) -> GlyphData[Outline]:
        return GlyphData(
            outline,
            sample.ref,
            {},
            codepoint=sample.codepoint,
            font_idx=0,
            character_idx=0,
            weight=400.0,
            width=100.0,
            italic=0.0,
            slant=0.0,
            optical_size=None,
        )

    first_data = build(Outline(types, coords))
    second_data = build(Outline(types.clone(), coords.clone()))

    assert first_data != second_data
    assert hash(first_data) == object.__hash__(first_data)


def test_glyph_data_targets_are_pytree_children() -> None:
    ref = GlyphRef(FontRef("font.ttf", 0), 36)
    location = {"wght": 400.0}
    data = GlyphData(
        torch.tensor([1.0]),
        ref,
        location,
        codepoint=0x41,
        font_idx=2,
        character_idx=3,
        weight=400.0,
        width=None,
        italic=0.0,
        slant=None,
        optical_size=12.0,
    )

    leaves, spec = pytree.tree_flatten(data)
    rebuilt = pytree.tree_unflatten(leaves, spec)

    assert leaves[1:] == [0x41, 2, 3, 400.0, None, 0.0, None, 12.0]
    assert rebuilt.ref is ref
    assert rebuilt.location is location
    assert rebuilt.codepoint == 0x41
    assert rebuilt.font_idx == 2
    assert rebuilt.width is None


def test_glyph_data_pytree_structure_does_not_depend_on_target_values() -> None:
    ref = GlyphRef(FontRef("font.ttf", 0), 36)
    location: dict[str, float] = {}

    def make(weight: float | None) -> GlyphData[torch.Tensor]:
        return GlyphData(
            torch.tensor([1.0]),
            ref,
            location,
            codepoint=0x41,
            font_idx=0,
            character_idx=0,
            weight=weight,
            width=None,
            italic=None,
            slant=None,
            optical_size=None,
        )

    _, first_spec = pytree.tree_flatten(make(400.0))
    _, second_spec = pytree.tree_flatten(make(None))

    assert first_spec == second_spec


def test_glyph_id_data_targets_are_pytree_children() -> None:
    ref = GlyphRef(FontRef("font.ttf", 0), 36)
    location = {"wght": 400.0}
    data = GlyphIdData(
        torch.tensor([1.0]),
        ref,
        location,
        font_idx=2,
        weight=400.0,
        width=None,
        italic=0.0,
        slant=None,
        optical_size=12.0,
    )

    leaves, spec = pytree.tree_flatten(data)
    rebuilt = pytree.tree_unflatten(leaves, spec)

    assert leaves[1:] == [2, 400.0, None, 0.0, None, 12.0]
    assert rebuilt.ref is ref
    assert rebuilt.location is location
    assert rebuilt.font_idx == 2
    assert rebuilt.width is None


def test_glyph_id_sample_identifies_face_and_glyph() -> None:
    sample = GlyphIdSample(GlyphRef(FontRef("font.ttf", 1), 36), 2)

    assert sample.ref.glyph_id == 36
    assert sample.font_idx == 2
