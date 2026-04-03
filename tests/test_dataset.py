from __future__ import annotations

import multiprocessing as mp
import pickle
import shutil
import subprocess
from pathlib import Path
from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

import torchfont
import torchfont.datasets as datasets_module
from torchfont.datasets import (
    DatasetMetadata,
    GlyphDataset,
    GlyphLocation,
    GlyphSample,
)
from torchfont.io import CommandType
from torchfont.metadata import build_dataset_metadata


def _read_first_sample_from_pickled_dataset(
    payload: bytes, queue: mp.Queue[tuple[int, int, int, tuple[int, int]]]
) -> None:
    dataset = pickle.loads(payload)  # noqa: S301
    sample = dataset[0]
    queue.put(
        (
            sample.style_idx,
            sample.content_idx,
            sample.types.numel(),
            tuple(sample.coords.shape),
        )
    )


def test_glyph_dataset_static_fonts() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=(
            "lato/Lato-Regular.ttf",
            "ubuntu/Ubuntu-Regular.ttf",
            "ptsans/PT_Sans-Web-Regular.ttf",
        ),
        codepoints=range(0x80),
    )

    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


def test_glyph_dataset_variable_fonts() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf", "notosansjp/NotoSansJP*.ttf"),
        codepoints=range(0x80),
    )

    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


def test_glyph_dataset_all_fonts() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoints=range(0x80),
    )

    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


def test_glyph_dataset_rejects_non_directory_root(tmp_path: Path) -> None:
    file_root = tmp_path / "not-a-directory.txt"
    file_root.write_text("not a font directory")

    with pytest.raises(ValueError, match="root must be a directory"):
        GlyphDataset(root=file_root)


def test_glyph_dataset_getitem() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    sample = dataset[0]

    assert isinstance(sample, GlyphSample)
    assert sample.types.dtype == torch.long
    assert sample.types.ndim == 1

    assert sample.coords.dtype == torch.float32
    assert sample.coords.ndim == 2
    assert sample.coords.shape[1] == 6
    assert isinstance(sample.style_idx, int)
    assert isinstance(sample.content_idx, int)
    assert 0 <= sample.style_idx < len(dataset.style_classes)
    assert 0 <= sample.content_idx < len(dataset.content_classes)


def test_glyph_dataset_locate_returns_source_metadata() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    sample = dataset[0]
    location = dataset.locate(0)

    assert isinstance(location, GlyphLocation)
    assert location.font_path == dataset.root / "lato/Lato-Regular.ttf"
    assert location.font_path.is_absolute()
    assert location.face_idx == 0
    assert location.instance_idx is None
    assert location.codepoint == ord("A")
    assert location.style_idx == sample.style_idx
    assert location.content_idx == sample.content_idx


def test_glyph_dataset_locate_tracks_variable_font_instance_index() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert len(dataset.style_classes) > 1
    codepoint_count = len(dataset.content_classes)

    first = dataset.locate(0)
    second = dataset.locate(codepoint_count)

    assert first.instance_idx == 0
    assert second.instance_idx == 1
    assert first.codepoint == ord("A")
    assert second.codepoint == ord("A")
    assert first.style_idx != second.style_idx


def test_datasets_public_api_is_glyphdataset_centered() -> None:
    assert datasets_module.__all__ == [
        "ContentLabel",
        "DatasetMetadata",
        "GlyphDataset",
        "GlyphLocation",
        "GlyphSample",
        "StyleLabel",
    ]
    assert datasets_module.DatasetMetadata is DatasetMetadata
    assert datasets_module.GlyphDataset is GlyphDataset
    assert datasets_module.GlyphLocation is GlyphLocation
    assert datasets_module.GlyphSample is GlyphSample
    assert not hasattr(datasets_module, "FontFolder")
    assert not hasattr(datasets_module, "FontRepo")
    assert not hasattr(datasets_module, "GoogleFonts")


def test_package_root_stays_thin() -> None:
    assert torchfont.__all__ == []
    assert not hasattr(torchfont, "GlyphDataset")
    assert not hasattr(torchfont, "GlyphSample")
    assert not hasattr(torchfont, "GlyphBatch")


def test_glyph_dataset_is_primary_local_api() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    sample = dataset[0]

    assert isinstance(dataset, GlyphDataset)
    assert isinstance(sample, GlyphSample)
    assert len(dataset) > 0


def test_glyph_dataset_transform_uses_sample_first_contract() -> None:
    calls: list[GlyphSample] = []

    def transform(sample: GlyphSample) -> GlyphSample:
        calls.append(sample)
        return GlyphSample(
            types=sample.types[:2],
            coords=sample.coords[:2],
            style_idx=sample.style_idx,
            content_idx=sample.content_idx,
        )

    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
        transform=transform,
    )
    sample = dataset[0]

    assert len(calls) == 1
    assert isinstance(calls[0], GlyphSample)
    assert sample.style_idx == calls[0].style_idx
    assert sample.content_idx == calls[0].content_idx
    assert sample.types.shape[0] == 2
    assert sample.coords.shape[0] == 2


def test_glyph_dataset_preserves_quadratic_curves() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[ord("o")],
    )

    sample = dataset[0]

    assert (sample.types == CommandType.QUAD_TO.value).any().item()
    assert not (sample.types == CommandType.CURVE_TO.value).any().item()


def test_glyph_dataset_negative_indexing() -> None:
    """Test that negative indexing works correctly."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    # Test that dataset[-1] returns the last element
    sample_last = dataset[-1]
    sample_explicit = dataset[len(dataset) - 1]

    # Verify that negative indexing returns the same result as positive indexing
    assert torch.equal(sample_last.types, sample_explicit.types)
    assert torch.equal(sample_last.coords, sample_explicit.coords)
    assert sample_last.style_idx == sample_explicit.style_idx
    assert sample_last.content_idx == sample_explicit.content_idx

    # Test dataset[-2] if dataset has at least 2 elements
    if len(dataset) >= 2:
        sample_sl = dataset[-2]
        sample_exp2 = dataset[len(dataset) - 2]

        assert torch.equal(sample_sl.types, sample_exp2.types)
        assert torch.equal(sample_sl.coords, sample_exp2.coords)
        assert sample_sl.style_idx == sample_exp2.style_idx
        assert sample_sl.content_idx == sample_exp2.content_idx


def test_glyph_dataset_index_out_of_bounds() -> None:
    """Test that out of bounds indices raise IndexError."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    # Test positive out of bounds
    with pytest.raises(IndexError):
        dataset[len(dataset)]

    with pytest.raises(IndexError):
        dataset[len(dataset) + 100]

    # Test negative out of bounds
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]

    with pytest.raises(IndexError):
        dataset[-len(dataset) - 100]


def test_glyph_dataset_locate_index_out_of_bounds() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    with pytest.raises(IndexError):
        dataset.locate(len(dataset))

    with pytest.raises(IndexError):
        dataset.locate(-len(dataset) - 1)


def test_glyph_dataset_cjk_support() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("notosansjp/NotoSansJP*.ttf",),
        codepoints=[ord(c) for c in "あいうえお"],
    )

    assert len(dataset) > 0
    sample = dataset[0]
    assert isinstance(sample, GlyphSample)
    assert sample.types.dtype == torch.long
    assert sample.types.ndim == 1
    assert sample.types.numel() > 0
    assert sample.coords.dtype == torch.float32
    assert sample.coords.ndim == 2
    assert sample.coords.shape[1] == 6
    assert isinstance(sample.style_idx, int)
    assert isinstance(sample.content_idx, int)
    assert 0 <= sample.style_idx < len(dataset.style_classes)
    assert 0 <= sample.content_idx < len(dataset.content_classes)


def test_glyph_dataset_skips_styles_without_samples_after_filtering() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf", "notosansjp/NotoSansJP*.ttf"),
        codepoints=[ord(c) for c in "あいう"],
    )

    assert len(dataset) > 0
    assert len(dataset.style_classes) == len(set(dataset.targets[:, 0].tolist()))
    assert sorted(set(dataset.targets[:, 0].tolist())) == list(
        range(len(dataset.style_classes))
    )
    assert all("Lato" not in name for name in dataset.style_classes)


def test_glyph_dataset_codepoints() -> None:
    dataset_upper = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    dataset_lower = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x61, 0x7B),
    )

    assert len(dataset_upper) > 0
    assert len(dataset_lower) > 0

    assert len(dataset_upper.content_classes) <= 26
    assert len(dataset_lower.content_classes) <= 26


def test_glyph_dataset_normalizes_codepoints_on_instance() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[ord("C"), ord("A"), ord("B"), ord("A")],
    )

    assert dataset.codepoints == (ord("A"), ord("B"), ord("C"))
    assert dataset.content_classes == ["A", "B", "C"]

    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301
    assert restored.codepoints == dataset.codepoints


@pytest.mark.parametrize(
    ("codepoints", "message"),
    [
        ([-1], "expected 0 <= cp <= 0x10FFFF"),
        ([0x110000], "expected 0 <= cp <= 0x10FFFF"),
        ([0xD800], "surrogate code points"),
    ],
)
def test_glyph_dataset_rejects_invalid_unicode_codepoints(
    codepoints: list[int], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        GlyphDataset(
            root="tests/fonts",
            patterns=("lato/Lato-Regular.ttf",),
            codepoints=codepoints,
        )


def test_glyph_dataset_pattern_filter() -> None:
    dataset_all = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoints=range(0x80),
    )

    dataset_roboto = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf", "notosansjp/NotoSans*.ttf"),
        codepoints=range(0x80),
    )

    dataset_lato = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x80),
    )

    assert len(dataset_all) > 0
    assert len(dataset_roboto) > 0
    assert len(dataset_lato) > 0
    assert len(dataset_all.style_classes) >= len(dataset_roboto.style_classes)
    assert len(dataset_all.style_classes) >= len(dataset_lato.style_classes)


def test_glyph_dataset_empty_result() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("nonexistent*.ttf",),
        codepoints=range(0x80),
    )
    assert len(dataset) == 0
    assert len(dataset.style_classes) == 0
    assert len(dataset.content_classes) == 0


def test_glyph_dataset_discovers_fonts_in_hidden_directories(
    tmp_path: Path,
) -> None:
    source = Path("tests/fonts/lato/Lato-Regular.ttf").resolve()
    hidden_dir = tmp_path / ".fonts"
    hidden_dir.mkdir()
    hidden_font = hidden_dir / "Lato-Regular.ttf"
    shutil.copy(source, hidden_font)

    dataset = GlyphDataset(root=tmp_path, codepoints=range(0x80))

    assert len(dataset) > 0
    assert dataset.locate(0).font_path == hidden_font.resolve()


def test_glyph_dataset_ignores_gitignore_for_root_discovery(tmp_path: Path) -> None:
    source = Path("tests/fonts/lato/Lato-Regular.ttf").resolve()
    git_executable = shutil.which("git")
    if git_executable is None:
        pytest.skip("git not installed")
    assert git_executable is not None
    subprocess.run(  # noqa: S603
        [git_executable, "init", "-q"],
        cwd=tmp_path,
        check=True,
    )
    font_path = tmp_path / "Lato-Regular.ttf"
    shutil.copy(source, font_path)
    (tmp_path / ".gitignore").write_text("*.ttf\n", encoding="utf-8")

    dataset = GlyphDataset(root=tmp_path, codepoints=range(0x80))

    assert len(dataset) > 0
    assert dataset.locate(0).font_path == font_path.resolve()


def test_glyph_dataset_skips_vcs_metadata_directories(tmp_path: Path) -> None:
    source = Path("tests/fonts/lato/Lato-Regular.ttf").resolve()
    vcs_dir = tmp_path / ".git"
    vcs_dir.mkdir()
    shutil.copy(source, vcs_dir / "Lato-Regular.ttf")

    dataset = GlyphDataset(root=tmp_path, codepoints=range(0x80))

    assert len(dataset) == 0
    assert dataset.style_classes == []
    assert dataset.content_classes == []


def test_content_classes() -> None:
    """Test content_classes returns Unicode character strings."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),  # A, B, C
    )

    assert len(dataset.content_classes) == 3
    assert dataset.content_classes == ["A", "B", "C"]
    assert all(isinstance(c, str) and len(c) == 1 for c in dataset.content_classes)


def test_content_class_to_idx() -> None:
    """Test content_class_to_idx maps characters to indices."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert dataset.content_class_to_idx["A"] == 0
    assert dataset.content_class_to_idx["B"] == 1
    assert dataset.content_class_to_idx["C"] == 2

    # Round-trip test
    for idx, char in enumerate(dataset.content_classes):
        assert dataset.content_class_to_idx[char] == idx


def test_content_classes_do_not_materialize_metadata() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    with patch.object(GlyphDataset, "metadata", new_callable=PropertyMock) as metadata:
        metadata.side_effect = AssertionError(
            "content_classes should not materialize metadata"
        )
        assert dataset.content_classes == ["A", "B", "C"]


def test_content_class_to_idx_does_not_materialize_metadata() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    with patch.object(GlyphDataset, "metadata", new_callable=PropertyMock) as metadata:
        metadata.side_effect = AssertionError(
            "content_class_to_idx should not materialize metadata"
        )
        assert dataset.content_class_to_idx == {"A": 0, "B": 1, "C": 2}


def test_style_classes() -> None:
    """Test style_classes returns descriptive names."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/*.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert len(dataset.style_classes) > 0
    assert all(isinstance(s, str) for s in dataset.style_classes)


def test_style_label_metadata_is_index_addressable() -> None:
    """Test metadata APIs are compatible with sample style/content indices."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    sample = dataset[0]
    style_label = dataset.style_labels[sample.style_idx]
    content_label = dataset.content_labels[sample.content_idx]

    assert style_label.idx == sample.style_idx
    assert style_label.label_id.startswith("style:path=")
    assert "instance=static" in style_label.label_id
    assert dataset.style_label_to_idx[style_label.label_id] == sample.style_idx
    assert sample.style_idx in dataset.style_name_to_idxs[style_label.name]

    assert content_label.idx == sample.content_idx
    assert dataset.content_label_to_idx[content_label.label_id] == sample.content_idx
    assert dataset.content_class_to_idx[content_label.char] == sample.content_idx


def test_dataset_metadata_consolidates_label_views() -> None:
    """DatasetMetadata provides a structured source of label metadata."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    metadata = dataset.metadata
    sample = dataset[0]

    assert isinstance(metadata, DatasetMetadata)
    assert metadata.styles[sample.style_idx] == dataset.style_labels[sample.style_idx]
    assert (
        metadata.contents[sample.content_idx]
        == dataset.content_labels[sample.content_idx]
    )
    assert metadata.style_id_to_idx == dataset.style_label_to_idx
    assert metadata.content_id_to_idx == dataset.content_label_to_idx
    assert dict(metadata.style_name_to_idxs) == {
        name: tuple(idxs) for name, idxs in dataset.style_name_to_idxs.items()
    }


def test_style_label_metadata_handles_duplicate_names() -> None:
    """Test duplicate style names are preserved in collision-safe metadata."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    raw_names = ["Shared", "Unique", "Shared"]
    with (
        patch.object(
            GlyphDataset,
            "style_classes",
            new_callable=PropertyMock,
            return_value=raw_names,
        ),
        patch.object(
            GlyphDataset,
            "_style_sources",
            return_value=[
                (dataset.root / "lato/Lato-Regular.ttf", 0, None),
                (dataset.root / "roboto/Roboto[wdth,wght].ttf", 0, 0),
                (dataset.root / "roboto/Roboto[wdth,wght].ttf", 0, 1),
            ],
        ),
    ):
        labels = dataset.style_labels
        grouped = dataset.style_name_to_idxs

    assert len({label.label_id for label in labels}) == 3
    assert grouped["Shared"] == [0, 2]
    assert grouped["Unique"] == [1]


def test_dataset_metadata_handles_duplicate_names() -> None:
    """DatasetMetadata preserves all indices for duplicate style names."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    raw_names = ["Shared", "Unique", "Shared"]
    with (
        patch.object(
            GlyphDataset,
            "style_classes",
            new_callable=PropertyMock,
            return_value=raw_names,
        ),
        patch.object(
            GlyphDataset,
            "_style_sources",
            return_value=[
                (dataset.root / "lato/Lato-Regular.ttf", 0, None),
                (dataset.root / "roboto/Roboto[wdth,wght].ttf", 0, 0),
                (dataset.root / "roboto/Roboto[wdth,wght].ttf", 0, 1),
            ],
        ),
    ):
        metadata = dataset.metadata

    assert len({label.label_id for label in metadata.styles}) == 3
    assert metadata.style_name_to_idxs["Shared"] == (0, 2)
    assert metadata.style_name_to_idxs["Unique"] == (1,)


def test_build_dataset_metadata_rejects_mismatched_style_inputs() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    with pytest.raises(
        ValueError,
        match="style_names and style_sources must have the same length",
    ):
        build_dataset_metadata(
            root=dataset.root,
            style_names=["Lato Regular"],
            style_sources=[],
            content_codepoints=[ord("A")],
        )


def test_style_label_ids_are_stable_across_codepoint_filters() -> None:
    dataset_a = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=range(0x41, 0x44),
    )
    dataset_b = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=range(0x41, 0x46),
    )

    assert [label.label_id for label in dataset_a.style_labels] == [
        label.label_id for label in dataset_b.style_labels
    ]


@pytest.mark.parametrize("start_method", [None, *mp.get_all_start_methods()])
def test_glyph_dataset_dataloader_multiworker(
    start_method: str | None,
) -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False,
        multiprocessing_context=start_method,
    )

    batch = next(iter(loader))
    assert batch is not None

    # DataLoader collates GlyphSample fields into a GlyphSample of batched tensors
    assert isinstance(batch, GlyphSample)

    # Validate types tensor (batch dimension added)
    assert batch.types.dtype == torch.long
    assert batch.types.ndim == 2  # batch_size x sequence_length

    # Validate coords tensor (batch dimension added)
    assert batch.coords.dtype == torch.float32
    assert batch.coords.ndim == 3  # batch_size x sequence_length x 6
    assert batch.coords.shape[2] == 6

    # Validate indices tensors
    assert batch.style_idx.dtype == torch.long
    assert batch.content_idx.dtype == torch.long
    assert batch.style_idx.ndim == 1  # batch_size
    assert batch.content_idx.ndim == 1  # batch_size

    # Validate index values are in valid range
    assert torch.all(batch.style_idx >= 0)
    assert torch.all(batch.style_idx < len(dataset.style_classes))
    assert torch.all(batch.content_idx >= 0)
    assert torch.all(batch.content_idx < len(dataset.content_classes))


def test_targets_shape_and_dtype() -> None:
    """Test that targets has shape (N, 2) and dtype long."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
    )

    assert dataset.targets.shape == (len(dataset), 2)
    assert dataset.targets.dtype == torch.long


def test_targets_matches_getitem() -> None:
    """Test that targets[i] matches the labels from __getitem__."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),  # A, B, C - small set
    )

    for i in range(len(dataset)):
        sample = dataset[i]
        assert dataset.targets[i, 0].item() == sample.style_idx
        assert dataset.targets[i, 1].item() == sample.content_idx


def test_targets_empty_dataset() -> None:
    """Test that targets has shape (0, 2) for an empty dataset."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("nonexistent*.ttf",),
        codepoints=range(0x80),
    )

    assert dataset.targets.shape == (0, 2)
    assert dataset.targets.dtype == torch.long


def test_targets_variable_fonts() -> None:
    """Test that targets is correct for variable fonts with multiple instances."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert len(dataset.style_classes) > 1
    assert dataset.targets.shape == (len(dataset), 2)
    assert dataset.targets.dtype == torch.long

    for i in range(len(dataset)):
        sample = dataset[i]
        assert dataset.targets[i, 0].item() == sample.style_idx
        assert dataset.targets[i, 1].item() == sample.content_idx


def test_glyph_dataset_repr() -> None:
    """Test that GlyphDataset has a useful __repr__."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    expected = (
        f"GlyphDataset("
        f"root={str(dataset.root)!r}, "
        f"samples={len(dataset)}, "
        f"styles={len(dataset.style_classes)}, "
        f"content_classes={len(dataset.content_classes)})"
    )
    assert repr(dataset) == expected


def test_glyph_dataset_repr_uses_native_count_getters() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )
    style_count = len(dataset.style_classes)
    content_count = len(dataset.content_classes)

    expected = (
        f"GlyphDataset("
        f"root={str(dataset.root)!r}, "
        f"samples={len(dataset)}, "
        f"styles={style_count}, "
        f"content_classes={content_count})"
    )

    with (
        patch.object(
            GlyphDataset, "style_classes", new_callable=PropertyMock
        ) as styles,
        patch.object(
            GlyphDataset,
            "content_classes",
            new_callable=PropertyMock,
        ) as contents,
    ):
        styles.side_effect = AssertionError("repr should not materialize style_classes")
        contents.side_effect = AssertionError(
            "repr should not materialize content_classes"
        )

        assert repr(dataset) == expected


def test_targets_survives_pickle() -> None:
    """Test that targets is correctly restored after pickle round-trip."""
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    original_targets = dataset.targets.clone()
    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301

    assert torch.equal(restored.targets, original_targets)


def test_glyph_dataset_getitem_survives_spawn_pickle_roundtrip() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    payload = pickle.dumps(dataset)
    ctx = mp.get_context("spawn")
    queue: mp.Queue[tuple[int, int, int, tuple[int, int]]] = ctx.Queue()
    proc = ctx.Process(
        target=_read_first_sample_from_pickled_dataset,
        args=(payload, queue),
    )
    proc.start()
    proc.join(timeout=30)

    assert proc.exitcode == 0
    style_idx, content_idx, types_len, coords_shape = queue.get(timeout=5)
    assert style_idx >= 0
    assert content_idx >= 0
    assert types_len > 0
    assert coords_shape[1] == 6


def test_glyph_dataset_filters_outline_less_glyphs() -> None:
    """Regression test for #61: outline-less glyphs must be excluded from the index.

    A font whose charmap maps codepoints to glyph IDs that have no outline data
    (e.g. color/bitmap-only fonts) previously caused len(dataset) > 0 while every
    dataset[i] raised ValueError. After the fix, such glyphs are filtered out at
    construction time so that len(dataset) == the number of items that can actually
    be retrieved via __getitem__.
    """
    # nocolortest/NoOutlines-Regular.ttf maps 'A' (U+0041) to a glyph with an
    # empty glyf table entry (zero-length loca slot), so outline_glyphs().get()
    # returns None for that glyph.
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("nocolortest/NoOutlines-Regular.ttf",),
        codepoints=range(0x80),
    )

    # All charmap'd glyphs have no outline data, so the dataset must be empty.
    assert len(dataset) == 0

    # Accessing an empty dataset must fail with IndexError rather than ValueError.
    with pytest.raises(IndexError):
        dataset[0]
