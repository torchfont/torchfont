from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch
from torch.utils.data import DataLoader

import torchfont
import torchfont.datasets as datasets_module
import torchfont.transforms as transforms_module
from tests._glyphs import glyph_id
from torchfont import (
    CodepointData,
    CodepointSample,
    ElementType,
    FontRef,
    GlyphRef,
    Outline,
    _torchfont,
)
from torchfont.datasets import CodepointDataset
from torchfont.transforms import LoadGlyph
from torchfont.transforms import functional as _functional

if TYPE_CHECKING:
    from collections.abc import Sequence


def _outline_pair(sample: CodepointSample) -> tuple[torch.Tensor, torch.Tensor]:
    outline = LoadGlyph()(sample).data
    return outline.types, outline.coords


def _worker_pair(sample: CodepointSample) -> tuple[torch.Tensor, torch.Tensor]:
    outline = LoadGlyph(location="random")(sample).data
    return outline.types, outline.coords


def test_dataset_indexes_each_face_codepoint_once() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns=("source-serif/SourceSerif4Variable-Roman.ttf",),
        codepoints=[0x41, 0x42],
    )

    assert len(dataset) == 2
    assert dataset.font_targets.tolist() == [0, 0]
    assert dataset.character_targets.tolist() == [0, 1]
    assert dataset.character_classes == ["A", "B"]
    sample = dataset[0]
    assert isinstance(sample, CodepointSample)
    assert isinstance(sample.ref, GlyphRef)


def test_dataset_treats_static_and_variable_files_as_one_face_each() -> None:
    variable = CodepointDataset(
        "tests/fonts",
        patterns="source-serif/SourceSerif4Variable-Roman.ttf",
        codepoints=[0x41],
    )
    static = CodepointDataset(
        "tests/fonts", patterns="source-sans/SourceSans3-Regular.ttf", codepoints=[0x41]
    )

    assert len(variable) == 1
    assert len(static) == 1


def test_dataset_transform_receives_sample() -> None:
    seen: list[CodepointSample] = []

    def transform(sample: CodepointSample) -> int:
        seen.append(sample)
        return sample.codepoint

    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=[0x41],
        transform=transform,
    )

    assert dataset[0] == 0x41
    assert len(seen) == 1


def test_load_glyph_returns_outline_tensors() -> None:
    sample = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=[ord("o")],
    )[0]

    outline = _functional.load_glyph(sample.ref)

    assert outline.types.dtype == torch.long
    assert outline.coords.dtype == torch.float32
    assert outline.coords.shape[1] == 6
    assert (outline.types == ElementType.QUAD_TO.value).any().item()
    assert not (outline.types == ElementType.CURVE_TO.value).any().item()


def test_dataset_resolves_glyph_ids_per_face() -> None:
    path = "static-collection/Metropolis.ttc"
    dataset = CodepointDataset("tests/fonts", patterns=path, codepoints=[ord("A")])

    resolved = [dataset[idx].ref.glyph_id for idx in range(len(dataset))]

    assert resolved == [
        glyph_id(f"tests/fonts/{path}", "A", ttc_index)
        for ttc_index in range(len(dataset.font_classes))
    ]


def test_load_glyph_reports_missing_glyph_id() -> None:
    path = "tests/fonts/source-sans/SourceSans3-Regular.ttf"
    ref = GlyphRef(FontRef(path, 0), 0xFFFF)

    with pytest.raises(IndexError, match="glyph id 65535 missing"):
        _functional.load_glyph(ref)


def test_dataset_reports_corrupt_font(tmp_path: Path) -> None:
    (tmp_path / "broken.ttf").write_bytes(b"not a font")

    with pytest.raises(ValueError, match="failed to parse"):
        CodepointDataset(tmp_path)


def test_load_glyph_reports_missing_font(tmp_path: Path) -> None:
    ref = GlyphRef(FontRef(str(tmp_path / "missing.ttf"), 0), 36)

    with pytest.raises(FileNotFoundError, match="failed to open"):
        _functional.load_glyph(ref)


def test_dataset_reports_invalid_pattern() -> None:
    with pytest.raises(ValueError, match="invalid pattern"):
        CodepointDataset("tests/fonts", patterns="[")


@pytest.mark.parametrize("codepoint", [1.5, "A"])
def test_dataset_rejects_non_integer_codepoints(codepoint: object) -> None:
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        CodepointDataset(
            "tests/fonts",
            patterns="source-sans/SourceSans3-Regular.ttf",
            codepoints=[codepoint],  # ty: ignore[invalid-argument-type]
        )


def test_explicit_location_validation() -> None:
    ref = CodepointDataset(
        "tests/fonts",
        patterns="source-serif/SourceSerif4Variable-Roman.ttf",
        codepoints=[0x41],
    )[0].ref

    with pytest.raises(ValueError, match="no variation axis 'xxxx'"):
        _functional.load_glyph(ref, {"xxxx": 1.0})
    with pytest.raises(ValueError, match="outside"):
        _functional.load_glyph(ref, {"wght": 10_000.0})
    with pytest.raises(ValueError, match="finite"):
        _functional.load_glyph(ref, {"wght": float("nan")})


def test_load_glyph_uses_default_location() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-serif/SourceSerif4Variable-Roman.ttf",
        codepoints=[0x41],
    )

    data = LoadGlyph()(dataset[0])

    assert isinstance(data, CodepointData)
    assert isinstance(data.data, Outline)
    assert data.location == {"opsz": 20.0, "wght": 400.0}


def test_load_glyph_samples_one_location_reproducibly() -> None:
    sample = CodepointDataset(
        "tests/fonts",
        patterns="source-serif/SourceSerif4Variable-Roman.ttf",
        codepoints=[0x41],
    )[0]

    torch.manual_seed(123)
    first = LoadGlyph(location="random")(sample)
    torch.manual_seed(123)
    second = LoadGlyph(location="random")(sample)

    assert first.location == second.location
    assert set(first.location) == {"opsz", "wght"}
    assert first.location != LoadGlyph()(sample).location


def test_load_glyph_resamples_location_on_each_call() -> None:
    sample = CodepointDataset(
        "tests/fonts",
        patterns="source-serif/SourceSerif4Variable-Roman.ttf",
        codepoints=[0x41],
    )[0]
    load = LoadGlyph(location="random")

    torch.manual_seed(123)
    first = load(sample)
    second = load(sample)

    assert first.location != second.location


def test_load_glyph_random_location_on_static_face_is_empty() -> None:
    sample = CodepointDataset(
        "tests/fonts", patterns="source-sans/SourceSans3-Regular.ttf", codepoints=[0x41]
    )[0]

    random = LoadGlyph(location="random")(sample)
    default = LoadGlyph()(sample)

    assert random.location == {}
    assert torch.equal(random.data.types, default.data.types)
    assert torch.equal(random.data.coords, default.data.coords)


def test_dataset_supports_multiprocessing_transform() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=[0x41, 0x42],
        transform=_outline_pair,
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


def test_load_glyph_random_location_supports_multiprocessing() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-serif/SourceSerif4Variable-Roman.ttf",
        codepoints=[0x41],
        transform=_worker_pair,
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
    dataset = CodepointDataset(
        "tests/fonts", patterns="source-sans/SourceSans3-Regular.ttf", codepoints=[0x41]
    )

    restored = cast(
        "CodepointDataset[CodepointSample]",
        pickle.loads(pickle.dumps(dataset)),  # noqa: S301
    )

    restored_sample = restored[0]
    sample = dataset[0]
    assert restored_sample.ref == sample.ref
    assert restored_sample.codepoint == sample.codepoint
    assert restored_sample.font_idx == sample.font_idx
    assert restored_sample.character_idx == sample.character_idx
    assert torch.equal(restored.font_targets, dataset.font_targets)
    assert torch.equal(restored.character_targets, dataset.character_targets)


@pytest.mark.parametrize("index", [-2, 1])
def test_dataset_rejects_out_of_range_indices(index: int) -> None:
    dataset = CodepointDataset(
        "tests/fonts", patterns="source-sans/SourceSans3-Regular.ttf", codepoints=[0x41]
    )
    with pytest.raises(IndexError):
        dataset[index]


def test_dataset_accepts_negative_index() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=[0x41, 0x42],
    )
    negative = dataset[-1]
    positive = dataset[1]
    assert negative.ref == positive.ref
    assert negative.codepoint == positive.codepoint
    assert negative.font_idx == positive.font_idx
    assert negative.character_idx == positive.character_idx


def test_dataset_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="failed to resolve font root"):
        CodepointDataset(tmp_path / "missing")


def test_dataset_ignores_fonts_without_requested_codepoints() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=[0x10FFFF],
    )
    assert len(dataset) == 0


def test_pattern_filter_and_outline_less_fonts_are_empty() -> None:
    missing = CodepointDataset("tests/fonts", patterns="nonexistent*.ttf")
    no_outlines = CodepointDataset(
        "tests/fonts", patterns="synthetic/NoOutlines-Regular.ttf"
    )

    assert len(missing) == 0
    assert missing.font_targets.shape == (0,)
    assert missing.character_targets.shape == (0,)
    assert len(no_outlines) == 0


def test_dataset_discovers_fonts_in_hidden_directories(tmp_path: Path) -> None:
    hidden = tmp_path / ".fonts"
    hidden.mkdir()
    shutil.copy(
        "tests/fonts/source-sans/SourceSans3-Regular.ttf",
        hidden / "SourceSans3-Regular.ttf",
    )

    dataset = CodepointDataset(tmp_path, codepoints=[0x41])

    assert len(dataset) == 1
    assert Path(dataset[0].ref.font.path).name == "SourceSans3-Regular.ttf"


def test_dataset_supports_non_utf8_font_paths(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows paths are Unicode")
    font_path = tmp_path / os.fsdecode(b"SourceSans-\xff.ttf")
    shutil.copy("tests/fonts/source-sans/SourceSans3-Regular.ttf", font_path)

    sample = CodepointDataset(tmp_path, codepoints=[0x41])[0]
    outline = _functional.load_glyph(sample.ref)

    assert "\udcff" in sample.ref.font.path
    assert outline.coords.shape[1] == 6


def test_dataset_ignores_gitignore_for_root_discovery(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)  # noqa: S603
    shutil.copy(
        "tests/fonts/source-sans/SourceSans3-Regular.ttf",
        tmp_path / "SourceSans3-Regular.ttf",
    )
    (tmp_path / ".gitignore").write_text("*.ttf\n", encoding="utf-8")

    dataset = CodepointDataset(tmp_path, codepoints=[0x41])

    assert len(dataset) == 1


def test_targets_match_samples() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=range(0x41, 0x44),
    )

    assert dataset.font_targets.dtype == torch.long
    assert dataset.character_targets.dtype == torch.long
    assert dataset.font_targets.shape == (len(dataset),)
    assert dataset.character_targets.shape == (len(dataset),)
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert dataset.font_targets[idx].item() == sample.font_idx
        assert dataset.character_targets[idx].item() == sample.character_idx


def test_targets_match_samples_across_faces_with_unequal_coverage() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        codepoints=[0x41, 0x42, 0xA9, 0x03A9, 0x0416, 0x2665, 0x3042, 0x4E00, 0xFB01],
    )
    font_targets = dataset.font_targets.tolist()
    character_targets = dataset.character_targets.tolist()
    character_class_to_idx = dataset.character_class_to_idx
    load = LoadGlyph()

    coverage: dict[int, tuple[int, ...]] = {}
    for idx in range(len(dataset)):
        sample = dataset[idx]
        assert font_targets[idx] == sample.font_idx
        assert character_targets[idx] == sample.character_idx
        assert character_class_to_idx[chr(sample.codepoint)] == sample.character_idx
        assert load(sample).data.types.numel() > 0
        coverage[sample.font_idx] = (
            *coverage.get(sample.font_idx, ()),
            sample.character_idx,
        )

    assert sorted(coverage) == list(range(len(dataset.font_classes)))
    assert any(indices != tuple(range(len(indices))) for indices in coverage.values())


def test_dataset_repr() -> None:
    dataset = CodepointDataset(
        "tests/fonts", patterns="source-sans/SourceSans3-Regular.ttf", codepoints=[0x41]
    )
    assert repr(dataset) == (
        f"CodepointDataset(root={str(dataset.root)!r}, samples=1, "
        "font_classes=1, character_classes=1)"
    )


def test_public_dataset_api_is_exported() -> None:
    assert datasets_module.CodepointDataset is CodepointDataset
    assert torchfont.CodepointSample is CodepointSample
    assert transforms_module.LoadGlyph is LoadGlyph
    assert hasattr(_torchfont, "index_codepoints")


def test_patterns_accept_string_or_sequence() -> None:
    string = CodepointDataset(
        "tests/fonts", patterns="source-sans/SourceSans3-Regular.ttf", codepoints=[0x41]
    )
    sequence = CodepointDataset(
        "tests/fonts",
        patterns=("source-sans/SourceSans3-Regular.ttf",),
        codepoints=[0x41],
    )
    string_sample = string[0]
    sequence_sample = sequence[0]
    assert string_sample.ref == sequence_sample.ref
    assert string_sample.codepoint == sequence_sample.codepoint
    assert string_sample.font_idx == sequence_sample.font_idx
    assert string_sample.character_idx == sequence_sample.character_idx


def test_codepoints_are_normalized_and_deduplicated() -> None:
    codepoints: Sequence[int] = [0x42, 0x41, 0x41]
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        codepoints=codepoints,
    )
    assert [dataset[i].codepoint for i in range(len(dataset))] == [0x41, 0x42]


def test_max_length_keeps_only_short_glyph_sequences() -> None:
    max_length = 12
    patterns = "source-sans/SourceSans3-Regular.ttf"
    codepoints = range(0x21, 0x7F)
    full = CodepointDataset(
        "tests/fonts",
        patterns=patterns,
        codepoints=codepoints,
        transform=LoadGlyph(),
    )
    expected = [
        (glyph.codepoint, glyph.ref.glyph_id)
        for glyph in (full[i] for i in range(len(full)))
        if glyph.data.num_elements <= max_length
    ]
    dataset = CodepointDataset(
        "tests/fonts",
        patterns=patterns,
        codepoints=codepoints,
        max_length=max_length,
        transform=LoadGlyph(),
    )

    assert 0 < len(expected) < len(full)
    assert [
        (glyph.codepoint, glyph.ref.glyph_id)
        for glyph in (dataset[i] for i in range(len(dataset)))
    ] == expected


def test_max_length_counts_the_end_element() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        max_length=1,
        transform=LoadGlyph(),
    )

    assert len(dataset) > 0
    assert all(
        dataset[i].data.types.tolist() == [ElementType.END] for i in range(len(dataset))
    )


def test_max_length_below_every_glyph_is_empty() -> None:
    dataset = CodepointDataset(
        "tests/fonts",
        patterns="source-sans/SourceSans3-Regular.ttf",
        max_length=0,
    )

    assert len(dataset) == 0
    assert dataset.font_classes == []
