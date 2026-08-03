import pickle

import pytest

from torchfont.structures import FontRef, GlyphRef, VariationLocation


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


def test_glyph_ref_normalizes_location() -> None:
    ref = GlyphRef(FontRef("font.ttf", 0), ord("A"), {"wght": 400})

    assert isinstance(ref.location, VariationLocation)
