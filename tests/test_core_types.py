import math
import pickle

import pytest
import torch

import torchfont
from torchfont import (
    COORD_DIM,
    TYPE_DIM,
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
    VariationLocation,
)


def test_core_types_are_exported_from_package_root() -> None:
    assert torchfont.COORD_DIM == COORD_DIM
    assert torchfont.TYPE_DIM == TYPE_DIM
    assert torchfont.ElementType is ElementType
    assert torchfont.FontRef is FontRef
    assert torchfont.GlyphData is GlyphData
    assert torchfont.GlyphRef is GlyphRef
    assert torchfont.GlyphSample is GlyphSample
    assert torchfont.Outline is Outline
    assert torchfont.VariationLocation is VariationLocation


def test_variation_location_is_order_independent_and_hashable() -> None:
    first = VariationLocation({"wght": 700, "wdth": 75})
    second = VariationLocation([("wdth", 75), ("wght", 700)])

    assert first == second
    assert first == {"wdth": 75.0, "wght": 700.0}
    assert hash(first) == hash(second)
    assert tuple(first) == ("wdth", "wght")


def test_variation_location_is_immutable() -> None:
    location = VariationLocation({"wght": 400})

    with pytest.raises(TypeError):
        location["wght"] = 700  # type: ignore[index]  # ty: ignore[invalid-assignment]


def test_variation_location_rejects_duplicate_normalized_tags() -> None:
    with pytest.raises(ValueError, match="duplicate variation axis tag 'wght'"):
        VariationLocation([("wght", 400), ("wght", 700)])


def test_variation_location_pickle_round_trip() -> None:
    location = VariationLocation({"wght": 400})

    assert pickle.loads(pickle.dumps(location)) == location  # noqa: S301


def test_glyph_ref_identifies_face_and_codepoint() -> None:
    ref = GlyphRef(FontRef("font.ttf", 0), ord("A"))

    assert ref.codepoint == ord("A")


@pytest.mark.parametrize(
    ("types", "coords", "match"),
    [
        (
            torch.zeros((1, 1), dtype=torch.long),
            torch.zeros((1, 6)),
            "types must be 1-D",
        ),
        (
            torch.zeros(1, dtype=torch.long),
            torch.zeros((1, 5)),
            "coords must have shape",
        ),
        (
            torch.zeros(1, dtype=torch.long),
            torch.zeros((2, 6)),
            "same number of rows",
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
    with pytest.raises(TypeError, match=r"coords must have dtype torch\.float32"):
        Outline(
            torch.zeros(1, dtype=torch.long), torch.zeros((1, 6), dtype=torch.int64)
        )


def test_outline_rejects_mismatched_devices() -> None:
    with pytest.raises(ValueError, match="must be on the same device"):
        Outline(
            torch.zeros(1, dtype=torch.long, device="meta"),
            torch.zeros((1, 6)),
        )


def test_semantic_containers_use_identity_equality() -> None:
    types = torch.tensor([1, 6], dtype=torch.long)
    coords = torch.zeros((2, 6))
    first = Outline(types, coords)
    second = Outline(types.clone(), coords.clone())
    sample = GlyphSample(GlyphRef(FontRef("font.ttf", 0), 0x41), 0, 0)
    first_data = GlyphData(
        first,
        sample.ref,
        VariationLocation(),
        0,
        0,
        400.0,
        100.0,
        0.0,
        0.0,
        math.nan,
    )
    second_data = GlyphData(
        second,
        sample.ref,
        VariationLocation(),
        0,
        0,
        400.0,
        100.0,
        0.0,
        0.0,
        math.nan,
    )

    assert first != second
    assert first_data != second_data
    assert hash(first) == object.__hash__(first)
