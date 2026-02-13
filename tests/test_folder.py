import multiprocessing as mp
import pickle
import warnings
from unittest.mock import PropertyMock, patch

import pytest
import torch
from torch.utils.data import DataLoader

from torchfont.datasets import FontFolder
from torchfont.io.outline import CommandType


def test_font_folder_static_fonts() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=(
            "lato/Lato-Regular.ttf",
            "ubuntu/Ubuntu-Regular.ttf",
            "ptsans/PT_Sans-Web-Regular.ttf",
        ),
        codepoint_filter=range(0x80),
    )

    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


def test_font_folder_variable_fonts() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf", "notosansjp/NotoSansJP*.ttf"),
        codepoint_filter=range(0x80),
    )

    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


def test_font_folder_all_fonts() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoint_filter=range(0x80),
    )

    assert len(dataset.style_classes) > 0
    assert len(dataset.content_classes) > 0
    assert len(dataset) > 0


def test_font_folder_getitem() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    types, coords, style_idx, content_idx = dataset[0]

    assert types.dtype == torch.long
    assert types.ndim == 1

    assert coords.dtype == torch.float32
    assert coords.ndim == 2
    assert coords.shape[1] == 6
    assert isinstance(style_idx, int)
    assert isinstance(content_idx, int)
    assert 0 <= style_idx < len(dataset.style_classes)
    assert 0 <= content_idx < len(dataset.content_classes)


def test_font_folder_preserves_quadratic_curves() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=[ord("o")],
    )

    types, _, _, _ = dataset[0]

    assert (types == CommandType.QUAD_TO.value).any().item()
    assert not (types == CommandType.CURVE_TO.value).any().item()


def test_font_folder_negative_indexing() -> None:
    """Test that negative indexing works correctly."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
    )

    assert len(dataset) > 0

    # Test that dataset[-1] returns the last element
    types_last, coords_last, style_last, content_last = dataset[-1]
    types_explicit, coords_explicit, style_explicit, content_explicit = dataset[
        len(dataset) - 1
    ]

    # Verify that negative indexing returns the same result as positive indexing
    assert torch.equal(types_last, types_explicit)
    assert torch.equal(coords_last, coords_explicit)
    assert style_last == style_explicit
    assert content_last == content_explicit

    # Test dataset[-2] if dataset has at least 2 elements
    if len(dataset) >= 2:
        types_second_last, coords_second_last, style_sl, content_sl = dataset[-2]
        types_explicit2, coords_explicit2, style_exp2, content_exp2 = dataset[
            len(dataset) - 2
        ]

        assert torch.equal(types_second_last, types_explicit2)
        assert torch.equal(coords_second_last, coords_explicit2)
        assert style_sl == style_exp2
        assert content_sl == content_exp2


def test_font_folder_index_out_of_bounds() -> None:
    """Test that out of bounds indices raise IndexError."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
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


def test_font_folder_cjk_support() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("notosansjp/NotoSansJP*.ttf",),
        codepoint_filter=[ord(c) for c in "あいうえお"],
    )

    assert len(dataset) > 0
    types, coords, style_idx, content_idx = dataset[0]
    assert types is not None
    assert coords is not None
    assert style_idx is not None
    assert content_idx is not None


def test_font_folder_codepoint_filter() -> None:
    dataset_upper = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
    )

    dataset_lower = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x61, 0x7B),
    )

    assert len(dataset_upper) > 0
    assert len(dataset_lower) > 0

    assert len(dataset_upper.content_classes) <= 26
    assert len(dataset_lower.content_classes) <= 26


def test_font_folder_pattern_filter() -> None:
    dataset_all = FontFolder(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoint_filter=range(0x80),
    )

    dataset_roboto = FontFolder(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf", "notosansjp/NotoSans*.ttf"),
        codepoint_filter=range(0x80),
    )

    dataset_lato = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x80),
    )

    assert len(dataset_all) > 0
    assert len(dataset_roboto) > 0
    assert len(dataset_lato) > 0
    assert len(dataset_all.style_classes) >= len(dataset_roboto.style_classes)
    assert len(dataset_all.style_classes) >= len(dataset_lato.style_classes)


def test_font_folder_empty_result() -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("nonexistent*.ttf",),
        codepoint_filter=range(0x80),
    )
    assert len(dataset) == 0
    assert len(dataset.style_classes) == 0
    assert len(dataset.content_classes) == 0


def test_content_classes() -> None:
    """Test content_classes returns Unicode character strings"""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),  # A, B, C
    )

    assert len(dataset.content_classes) == 3
    assert dataset.content_classes == ["A", "B", "C"]
    assert all(isinstance(c, str) and len(c) == 1 for c in dataset.content_classes)


def test_content_class_to_idx() -> None:
    """Test content_class_to_idx maps characters to indices"""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    assert dataset.content_class_to_idx["A"] == 0
    assert dataset.content_class_to_idx["B"] == 1
    assert dataset.content_class_to_idx["C"] == 2

    # Round-trip test
    for idx, char in enumerate(dataset.content_classes):
        assert dataset.content_class_to_idx[char] == idx


def test_style_classes() -> None:
    """Test style_classes returns descriptive names"""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/*.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    assert len(dataset.style_classes) > 0
    assert all(isinstance(s, str) for s in dataset.style_classes)


def test_style_class_to_idx() -> None:
    """Test style_class_to_idx maps names to indices"""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/*.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    # Round-trip test
    for idx, name in enumerate(dataset.style_classes):
        assert dataset.style_class_to_idx[name] == idx


def test_style_class_to_idx_warns_on_duplicates() -> None:
    """Test that style_class_to_idx emits a UserWarning on duplicate names."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    raw_names = ["Roboto Regular", "Roboto Bold", "Roboto Regular"]
    with patch.object(
        FontFolder,
        "style_classes",
        new_callable=PropertyMock,
        return_value=raw_names,
    ):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            mapping = dataset.style_class_to_idx
            assert len(w) == 1
            assert issubclass(w[0].category, UserWarning)
            assert "Roboto Regular" in str(w[0].message)

        assert len(mapping) == 2  # duplicates collapsed
        assert mapping["Roboto Regular"] == 2  # keeps last occurrence
        assert mapping["Roboto Bold"] == 1


def test_style_class_to_idx_no_warning_without_duplicates() -> None:
    """Test that no warning is emitted when style names are unique."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        _ = dataset.style_class_to_idx
        assert len(w) == 0


@pytest.mark.parametrize("start_method", [None, *mp.get_all_start_methods()])
def test_font_folder_dataloader_multiworker(
    start_method: str | None,
) -> None:
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
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

    # Unpack batch and validate structure similar to test_font_folder_getitem
    types_batch, coords_batch, style_idx_batch, content_idx_batch = batch

    # Validate types tensor (batch dimension added)
    assert types_batch.dtype == torch.long
    assert types_batch.ndim == 2  # batch_size x sequence_length

    # Validate coords tensor (batch dimension added)
    assert coords_batch.dtype == torch.float32
    assert coords_batch.ndim == 3  # batch_size x sequence_length x 6
    assert coords_batch.shape[2] == 6

    # Validate indices tensors
    assert style_idx_batch.dtype == torch.long
    assert content_idx_batch.dtype == torch.long
    assert style_idx_batch.ndim == 1  # batch_size
    assert content_idx_batch.ndim == 1  # batch_size

    # Validate index values are in valid range
    assert torch.all(style_idx_batch >= 0)
    assert torch.all(style_idx_batch < len(dataset.style_classes))
    assert torch.all(content_idx_batch >= 0)
    assert torch.all(content_idx_batch < len(dataset.content_classes))


def test_targets_shape_and_dtype() -> None:
    """Test that targets has shape (N, 2) and dtype long."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x5B),
    )

    assert dataset.targets.shape == (len(dataset), 2)
    assert dataset.targets.dtype == torch.long


def test_targets_matches_getitem() -> None:
    """Test that targets[i] matches the labels from __getitem__."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),  # A, B, C — small set
    )

    for i in range(len(dataset)):
        _, _, style_idx, content_idx = dataset[i]
        assert dataset.targets[i, 0].item() == style_idx
        assert dataset.targets[i, 1].item() == content_idx


def test_targets_empty_dataset() -> None:
    """Test that targets has shape (0, 2) for an empty dataset."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("nonexistent*.ttf",),
        codepoint_filter=range(0x80),
    )

    assert dataset.targets.shape == (0, 2)
    assert dataset.targets.dtype == torch.long


def test_targets_variable_fonts() -> None:
    """Test that targets is correct for variable fonts with multiple instances."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    assert len(dataset.style_classes) > 1
    assert dataset.targets.shape == (len(dataset), 2)
    assert dataset.targets.dtype == torch.long

    for i in range(len(dataset)):
        _, _, style_idx, content_idx = dataset[i]
        assert dataset.targets[i, 0].item() == style_idx
        assert dataset.targets[i, 1].item() == content_idx


def test_targets_survives_pickle() -> None:
    """Test that targets is correctly restored after pickle round-trip."""
    dataset = FontFolder(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoint_filter=range(0x41, 0x44),
    )

    original_targets = dataset.targets.clone()
    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301

    assert torch.equal(restored.targets, original_targets)
