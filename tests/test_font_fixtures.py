import pytest

from torchfont import ElementType
from torchfont.datasets import CodepointDataset
from torchfont.transforms import LoadGlyph

FONT_ROOT = "tests/fonts"


@pytest.mark.parametrize(
    ("path", "curve_type", "location_keys"),
    [
        (
            "source-sans/SourceSans3-Regular.ttf",
            ElementType.QUAD_TO,
            set(),
        ),
        (
            "source-sans/SourceSans3-Regular.otf",
            ElementType.CURVE_TO,
            set(),
        ),
        (
            "source-serif/SourceSerif4Variable-Roman.ttf",
            ElementType.QUAD_TO,
            {"opsz", "wght"},
        ),
        (
            "source-serif/SourceSerif4Variable-Roman.otf",
            ElementType.CURVE_TO,
            {"opsz", "wght"},
        ),
    ],
)
def test_single_font_fixtures_cover_outline_and_variation_formats(
    path: str,
    curve_type: ElementType,
    location_keys: set[str],
) -> None:
    dataset = CodepointDataset(FONT_ROOT, patterns=path, codepoints=[ord("o")])

    assert len(dataset) == 1
    glyph = LoadGlyph()(dataset[0])
    assert (glyph.data.types == curve_type.value).any().item()
    assert set(glyph.location) == location_keys


@pytest.mark.parametrize(
    ("path", "codepoint", "face_count", "location_keys"),
    [
        ("static-collection/Metropolis.ttc", ord("A"), 19, set()),
        (
            "variable-collection/SourceHanSansVFProto.ttc",
            ord("A"),
            6,
            {"wdth", "wght"},
        ),
    ],
)
def test_collection_fixtures_expose_every_face(
    path: str,
    codepoint: int,
    face_count: int,
    location_keys: set[str],
) -> None:
    dataset = CodepointDataset(FONT_ROOT, patterns=path, codepoints=[codepoint])

    assert len(dataset) == face_count
    assert [font.face_index for font in dataset.font_classes] == list(range(face_count))
    for sample in dataset:
        glyph = LoadGlyph()(sample)
        assert glyph.data.types.numel() > 0
        assert set(glyph.location) == location_keys
