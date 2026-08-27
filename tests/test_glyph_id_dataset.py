from __future__ import annotations

import multiprocessing as mp
import pickle
from pathlib import PurePath
from typing import TYPE_CHECKING, cast

import pytest
import torch
from fontTools.ttLib import TTFont
from torch.utils.data import DataLoader

import torchfont
import torchfont.datasets as datasets_module
from tests._glyphs import glyph_id_by_name
from torchfont import GlyphIdData, GlyphIdSample, GlyphRef, Outline, _torchfont
from torchfont.datasets import CodepointDataset, GlyphIdDataset
from torchfont.transforms import LoadGlyph
from torchfont.transforms import functional as _functional

if TYPE_CHECKING:
    from pathlib import Path

STATIC_FONT = "source-sans/SourceSans3-Regular.ttf"
VARIABLE_FONT = "source-serif/SourceSerif4Variable-Roman.ttf"
COLLECTION_FONT = "static-collection/Metropolis.ttc"


def _outline_pair(sample: GlyphIdSample) -> tuple[torch.Tensor, torch.Tensor]:
    outline = LoadGlyph()(sample).data
    return outline.types, outline.coords


def test_dataset_indexes_every_outline_glyph_of_each_face() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT)
    font = TTFont(f"tests/fonts/{STATIC_FONT}")

    assert len(dataset) == font["maxp"].numGlyphs
    assert dataset.glyph_ids.tolist() == list(range(len(dataset)))
    assert dataset.outline_lengths.dtype == torch.long
    assert dataset.outline_lengths.shape == (len(dataset),)
    assert dataset.outline_lengths.tolist() == [
        len(LoadGlyph()(dataset[idx]).data) for idx in range(len(dataset))
    ]
    sample = dataset[0]
    assert isinstance(sample, GlyphIdSample)
    assert isinstance(sample.ref, GlyphRef)
    assert sample.ref.glyph_id == 0
    assert sample.font_idx == 0


def test_dataset_reaches_glyphs_no_codepoint_maps_to() -> None:
    path = f"tests/fonts/{STATIC_FONT}"
    ligature_id = glyph_id_by_name(path, "f_f_t")
    dataset = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT)
    codepoint_dataset = CodepointDataset("tests/fonts", patterns=STATIC_FONT)

    mapped = {
        codepoint_dataset[idx].ref.glyph_id for idx in range(len(codepoint_dataset))
    }
    assert ligature_id not in mapped
    sample = dataset[ligature_id]

    assert sample.ref.glyph_id == ligature_id
    outline = _functional.load_glyph(sample.ref)
    assert outline.types.numel() > 0


def test_dataset_indexes_each_face_of_a_collection() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=COLLECTION_FONT)

    assert [ref.face_index for ref in dataset.font_classes] == list(
        range(len(dataset.font_classes))
    )
    assert {PurePath(ref.path).name for ref in dataset.font_classes} == {
        "Metropolis.ttc"
    }
    per_face = dataset.font_targets.bincount()
    assert per_face.tolist() == [
        TTFont(f"tests/fonts/{COLLECTION_FONT}", fontNumber=idx)["maxp"].numGlyphs
        for idx in range(len(dataset.font_classes))
    ]


def test_targets_match_samples() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=COLLECTION_FONT)

    assert dataset.font_targets.dtype == torch.long
    assert dataset.glyph_ids.dtype == torch.long
    assert dataset.font_targets.shape == (len(dataset),)
    assert dataset.glyph_ids.shape == (len(dataset),)
    for idx in (0, len(dataset) // 2, len(dataset) - 1):
        sample = dataset[idx]
        assert dataset.font_targets[idx].item() == sample.font_idx
        assert dataset.glyph_ids[idx].item() == sample.ref.glyph_id
        assert dataset.font_classes[sample.font_idx] == sample.ref.font


def test_dataset_transform_receives_sample() -> None:
    seen: list[GlyphIdSample] = []

    def transform(sample: GlyphIdSample) -> int:
        seen.append(sample)
        return sample.ref.glyph_id

    dataset = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT, transform=transform)

    assert dataset[7] == 7
    assert len(seen) == 1


def test_load_glyph_returns_glyph_id_data() -> None:
    sample = GlyphIdDataset("tests/fonts", patterns=VARIABLE_FONT)[
        glyph_id_by_name(f"tests/fonts/{VARIABLE_FONT}", "f_i")
    ]

    data = LoadGlyph()(sample)

    assert isinstance(data, GlyphIdData)
    assert isinstance(data.data, Outline)
    assert data.ref == sample.ref
    assert data.font_idx == sample.font_idx
    assert data.location == {"opsz": 20.0, "wght": 400.0}
    assert data.weight == 400.0
    assert not hasattr(data, "codepoint")


def test_load_glyph_samples_one_location_reproducibly() -> None:
    sample = GlyphIdDataset("tests/fonts", patterns=VARIABLE_FONT)[0]

    torch.manual_seed(123)
    first = LoadGlyph(location="random")(sample)
    torch.manual_seed(123)
    second = LoadGlyph(location="random")(sample)

    assert first.location == second.location
    assert set(first.location) == {"opsz", "wght"}


def test_dataset_supports_multiprocessing_transform() -> None:
    dataset = GlyphIdDataset(
        "tests/fonts", patterns=STATIC_FONT, transform=_outline_pair
    )
    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=1,
        multiprocessing_context=mp.get_context("spawn"),
    )

    types, coords = next(iter(loader))
    assert types.shape[0] == 1
    assert coords.shape[0] == 1


def test_dataset_is_pickleable() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT)

    restored = cast(
        "GlyphIdDataset[GlyphIdSample]",
        pickle.loads(pickle.dumps(dataset)),  # noqa: S301
    )

    assert restored[36] == dataset[36]
    assert torch.equal(restored.font_targets, dataset.font_targets)
    assert torch.equal(restored.glyph_ids, dataset.glyph_ids)
    assert torch.equal(restored.outline_lengths, dataset.outline_lengths)


def test_dataset_accepts_negative_index() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT)

    assert dataset[-1] == dataset[len(dataset) - 1]


@pytest.mark.parametrize("index", [-1, 0])
def test_dataset_rejects_out_of_range_indices(index: int) -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns="nonexistent*.ttf")

    with pytest.raises(IndexError):
        dataset[index]


def test_pattern_filter_and_outline_less_fonts_are_empty() -> None:
    missing = GlyphIdDataset("tests/fonts", patterns="nonexistent*.ttf")
    no_outlines = GlyphIdDataset(
        "tests/fonts", patterns="synthetic/NoOutlines-Regular.ttf"
    )

    assert len(missing) == 0
    assert missing.font_targets.shape == (0,)
    assert missing.glyph_ids.shape == (0,)
    assert missing.outline_lengths.shape == (0,)
    assert len(no_outlines) == 0
    assert no_outlines.font_classes == []


def test_dataset_reports_invalid_pattern() -> None:
    with pytest.raises(ValueError, match="invalid pattern"):
        GlyphIdDataset("tests/fonts", patterns="[")


def test_dataset_reports_corrupt_font(tmp_path: Path) -> None:
    (tmp_path / "broken.ttf").write_bytes(b"not a font")

    with pytest.raises(ValueError, match="failed to parse"):
        GlyphIdDataset(tmp_path)


def test_dataset_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="failed to resolve font root"):
        GlyphIdDataset(tmp_path / "missing")


def test_dataset_repr() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=VARIABLE_FONT)

    assert repr(dataset) == (
        f"GlyphIdDataset(root={str(dataset.root)!r}, samples={len(dataset)}, "
        "font_classes=1)"
    )


def test_public_dataset_api_is_exported() -> None:
    assert datasets_module.GlyphIdDataset is GlyphIdDataset
    assert torchfont.GlyphIdSample is GlyphIdSample
    assert torchfont.GlyphIdData is GlyphIdData
    assert hasattr(_torchfont, "index_glyphs")


def test_max_length_keeps_only_short_glyph_sequences() -> None:
    max_length = 12
    full = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT, transform=LoadGlyph())
    expected = [
        glyph.ref.glyph_id
        for glyph in (full[i] for i in range(len(full)))
        if glyph.data.num_elements <= max_length
    ]
    dataset = GlyphIdDataset(
        "tests/fonts",
        patterns=STATIC_FONT,
        max_length=max_length,
        transform=LoadGlyph(),
    )

    assert 0 < len(expected) < len(full)
    assert dataset.glyph_ids.tolist() == expected
    assert all(dataset[i].data.num_elements <= max_length for i in range(len(dataset)))


def test_max_length_below_every_glyph_is_empty() -> None:
    dataset = GlyphIdDataset("tests/fonts", patterns=STATIC_FONT, max_length=0)

    assert len(dataset) == 0
    assert dataset.font_classes == []
