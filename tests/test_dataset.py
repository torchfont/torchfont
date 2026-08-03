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

import torchfont.datasets as datasets_module
import torchfont.structures as structures_module
import torchfont.transforms as transforms_module
from torchfont import _torchfont
from torchfont.datasets import GlyphDataset
from torchfont.structures import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
)
from torchfont.transforms import LoadGlyph
from torchfont.transforms import functional as _functional

if TYPE_CHECKING:
    from collections.abc import Sequence


def _outline_pair(sample: GlyphSample) -> tuple[torch.Tensor, torch.Tensor]:
    outline = LoadGlyph()(sample).data
    return outline.types, outline.coords


def _worker_pair(sample: GlyphSample) -> tuple[torch.Tensor, torch.Tensor]:
    outline = LoadGlyph(location="random")(sample).data
    return outline.types, outline.coords


def test_dataset_indexes_each_face_codepoint_once() -> None:
    dataset = GlyphDataset(
        "tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41, 0x42],
    )

    assert len(dataset) == 2
    assert dataset.font_targets.tolist() == [0, 0]
    assert dataset.character_targets.tolist() == [0, 1]
    assert dataset.character_classes == ["A", "B"]
    assert dataset[0] == dataset[0]
    assert isinstance(dataset[0], GlyphSample)
    assert isinstance(dataset[0].ref, GlyphRef)


def test_dataset_treats_static_and_variable_files_as_one_face_each() -> None:
    variable = GlyphDataset(
        "tests/fonts", patterns="roboto/Roboto*.ttf", codepoints=[0x41]
    )
    static = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x41]
    )

    assert len(variable) == 1
    assert len(static) == 1


def test_dataset_transform_receives_sample() -> None:
    seen: list[GlyphSample] = []

    def transform(sample: GlyphSample) -> int:
        seen.append(sample)
        return sample.ref.codepoint

    dataset = GlyphDataset(
        "tests/fonts",
        patterns="lato/Lato-Regular.ttf",
        codepoints=[0x41],
        transform=transform,
    )

    assert dataset[0] == 0x41
    assert len(seen) == 1


def test_load_glyph_returns_outline_tensors() -> None:
    sample = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[ord("o")]
    )[0]

    outline = _functional.load_glyph(sample.ref)

    assert outline.types.dtype == torch.long
    assert outline.coords.dtype == torch.float32
    assert outline.coords.shape[1] == 6
    assert (outline.types == ElementType.QUAD_TO.value).any().item()
    assert not (outline.types == ElementType.CURVE_TO.value).any().item()


def test_dataset_reports_corrupt_font(tmp_path: Path) -> None:
    (tmp_path / "broken.ttf").write_bytes(b"not a font")

    with pytest.raises(ValueError, match="failed to parse"):
        GlyphDataset(tmp_path)


def test_load_glyph_reports_missing_font(tmp_path: Path) -> None:
    ref = GlyphRef(FontRef(str(tmp_path / "missing.ttf"), 0), ord("A"))

    with pytest.raises(FileNotFoundError, match="failed to open"):
        _functional.load_glyph(ref)


def test_dataset_reports_invalid_pattern() -> None:
    with pytest.raises(ValueError, match="invalid pattern"):
        GlyphDataset("tests/fonts", patterns="[")


@pytest.mark.parametrize("codepoint", [1.5, "A"])
def test_dataset_rejects_non_integer_codepoints(codepoint: object) -> None:
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        GlyphDataset(
            "tests/fonts",
            patterns="lato/Lato-Regular.ttf",
            codepoints=[codepoint],  # ty: ignore[invalid-argument-type]
        )


def test_explicit_location_validation() -> None:
    ref = GlyphDataset("tests/fonts", patterns="roboto/Roboto*.ttf", codepoints=[0x41])[
        0
    ].ref

    with pytest.raises(ValueError, match="no variation axis 'xxxx'"):
        _functional.load_glyph(ref, {"xxxx": 1.0})
    with pytest.raises(ValueError, match="outside"):
        _functional.load_glyph(ref, {"wght": 10_000.0})
    with pytest.raises(ValueError, match="finite"):
        _functional.load_glyph(ref, {"wght": float("nan")})


def test_load_glyph_uses_default_location() -> None:
    dataset = GlyphDataset(
        "tests/fonts", patterns="roboto/Roboto*.ttf", codepoints=[0x41]
    )

    data = LoadGlyph()(dataset[0])

    assert isinstance(data, GlyphData)
    assert isinstance(data.data, Outline)
    assert data.location == {"wdth": 100.0, "wght": 400.0}


def test_load_glyph_samples_one_location_reproducibly() -> None:
    sample = GlyphDataset(
        "tests/fonts", patterns="roboto/Roboto*.ttf", codepoints=[0x41]
    )[0]

    torch.manual_seed(123)
    first = LoadGlyph(location="random")(sample)
    torch.manual_seed(123)
    second = LoadGlyph(location="random")(sample)

    assert first.location == second.location
    assert set(first.location) == {"wdth", "wght"}
    assert first.location != LoadGlyph()(sample).location


def test_load_glyph_random_location_on_static_face_is_empty() -> None:
    sample = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x41]
    )[0]

    random = LoadGlyph(location="random")(sample)
    default = LoadGlyph()(sample)

    assert random.location == {}
    assert torch.equal(random.data.types, default.data.types)
    assert torch.equal(random.data.coords, default.data.coords)


def test_dataset_supports_multiprocessing_transform() -> None:
    dataset = GlyphDataset(
        "tests/fonts",
        patterns="lato/Lato-Regular.ttf",
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
    dataset = GlyphDataset(
        "tests/fonts",
        patterns="roboto/Roboto*.ttf",
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
    dataset = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x41]
    )

    restored = cast(
        "GlyphDataset[GlyphSample]",
        pickle.loads(pickle.dumps(dataset)),  # noqa: S301
    )

    assert restored[0] == dataset[0]
    assert torch.equal(restored.font_targets, dataset.font_targets)
    assert torch.equal(restored.character_targets, dataset.character_targets)


@pytest.mark.parametrize("index", [-2, 1])
def test_dataset_rejects_out_of_range_indices(index: int) -> None:
    dataset = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x41]
    )
    with pytest.raises(IndexError):
        dataset[index]


def test_dataset_accepts_negative_index() -> None:
    dataset = GlyphDataset(
        "tests/fonts",
        patterns="lato/Lato-Regular.ttf",
        codepoints=[0x41, 0x42],
    )
    assert dataset[-1] == dataset[1]


def test_dataset_rejects_missing_root(tmp_path: Path) -> None:
    with pytest.raises(OSError, match="failed to resolve font root"):
        GlyphDataset(tmp_path / "missing")


def test_dataset_ignores_fonts_without_requested_codepoints() -> None:
    dataset = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x10FFFF]
    )
    assert len(dataset) == 0


def test_pattern_filter_and_outline_less_fonts_are_empty() -> None:
    missing = GlyphDataset("tests/fonts", patterns="nonexistent*.ttf")
    no_outlines = GlyphDataset(
        "tests/fonts", patterns="nocolortest/NoOutlines-Regular.ttf"
    )

    assert len(missing) == 0
    assert missing.font_targets.shape == (0,)
    assert missing.character_targets.shape == (0,)
    assert len(no_outlines) == 0


def test_dataset_discovers_fonts_in_hidden_directories(tmp_path: Path) -> None:
    hidden = tmp_path / ".fonts"
    hidden.mkdir()
    shutil.copy("tests/fonts/lato/Lato-Regular.ttf", hidden / "Lato-Regular.ttf")

    dataset = GlyphDataset(tmp_path, codepoints=[0x41])

    assert len(dataset) == 1
    assert Path(dataset[0].ref.font.path).name == "Lato-Regular.ttf"


def test_dataset_supports_non_utf8_font_paths(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows paths are Unicode")
    font_path = tmp_path / os.fsdecode(b"Lato-\xff.ttf")
    shutil.copy("tests/fonts/lato/Lato-Regular.ttf", font_path)

    sample = GlyphDataset(tmp_path, codepoints=[0x41])[0]
    outline = _functional.load_glyph(sample.ref)

    assert "\udcff" in sample.ref.font.path
    assert outline.coords.shape[1] == 6


def test_dataset_ignores_gitignore_for_root_discovery(tmp_path: Path) -> None:
    git = shutil.which("git")
    if git is None:
        pytest.skip("git not installed")
    subprocess.run([git, "init", "-q"], cwd=tmp_path, check=True)  # noqa: S603
    shutil.copy("tests/fonts/lato/Lato-Regular.ttf", tmp_path / "Lato-Regular.ttf")
    (tmp_path / ".gitignore").write_text("*.ttf\n", encoding="utf-8")

    dataset = GlyphDataset(tmp_path, codepoints=[0x41])

    assert len(dataset) == 1


def test_targets_match_samples() -> None:
    dataset = GlyphDataset(
        "tests/fonts",
        patterns="lato/Lato-Regular.ttf",
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


def test_dataset_repr() -> None:
    dataset = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x41]
    )
    assert repr(dataset) == (
        f"GlyphDataset(root={str(dataset.root)!r}, samples=1, "
        "font_classes=1, character_classes=1)"
    )


def test_public_dataset_api_is_exported() -> None:
    assert datasets_module.GlyphDataset is GlyphDataset
    assert structures_module.GlyphSample is GlyphSample
    assert transforms_module.LoadGlyph is LoadGlyph
    assert hasattr(_torchfont, "GlyphIndex")


def test_patterns_accept_string_or_sequence() -> None:
    string = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=[0x41]
    )
    sequence = GlyphDataset(
        "tests/fonts", patterns=("lato/Lato-Regular.ttf",), codepoints=[0x41]
    )
    assert string[0] == sequence[0]


def test_codepoints_are_normalized_and_deduplicated() -> None:
    codepoints: Sequence[int] = [0x42, 0x41, 0x41]
    dataset = GlyphDataset(
        "tests/fonts", patterns="lato/Lato-Regular.ttf", codepoints=codepoints
    )
    assert [dataset[i].ref.codepoint for i in range(len(dataset))] == [0x41, 0x42]
