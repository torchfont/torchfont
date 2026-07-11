import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import cast

import pytest
import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torchfont
import torchfont.datasets as datasets_module
import torchfont.transforms as transforms_module
import torchfont.variation as variation_module
from torchfont import _torchfont
from torchfont.datasets import (
    FontRef,
    GlyphDataset,
    GlyphRef,
    GlyphSample,
    VariableGlyphDataset,
    VariableGlyphRef,
    VariableGlyphSample,
)
from torchfont.io import ElementType
from torchfont.transforms import load_glyph
from torchfont.variation import (
    default_instance,
    default_instance_count,
    grid_instance_count,
    grid_instances,
    named_instance_count,
    named_instances,
    random_location,
)


def _to_pair(sample: GlyphSample) -> tuple[torch.Tensor, torch.Tensor]:
    return load_glyph(sample.ref)


def _variable_to_pair(
    sample: VariableGlyphSample,
) -> tuple[torch.Tensor, torch.Tensor]:
    location = random_location(sample.ref.font)
    return load_glyph(sample.ref, location)


def _collate_outline(
    batch: list[tuple[torch.Tensor, torch.Tensor]],
) -> tuple[torch.Tensor, torch.Tensor]:
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


def _read_first_sample_from_pickled_dataset(
    payload: bytes,
    queue: mp.Queue[tuple[int, int, int, tuple[int, int]]],
) -> None:
    dataset = cast("GlyphDataset[GlyphSample]", pickle.loads(payload))  # noqa: S301
    sample = dataset[0]
    types, coords = load_glyph(sample.ref)
    coords_shape = (int(coords.shape[0]), int(coords.shape[1]))
    queue.put(
        (
            sample.font_idx,
            sample.character_idx,
            int(types.numel()),
            coords_shape,
        ),
    )


def test_glyph_dataset_static_fonts_returns_refs() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert repr(dataset) == (
        f"GlyphDataset(root={str(dataset.root)!r}, samples=3, "
        "font_classes=1, styles=1, character_classes=3)"
    )
    assert len(dataset.font_classes) == 1
    assert dataset.character_classes == ["A", "B", "C"]
    assert dataset.character_class_to_idx == {"A": 0, "B": 1, "C": 2}

    sample = dataset[0]

    assert isinstance(sample, GlyphSample)
    assert isinstance(sample.ref, GlyphRef)
    assert sample.ref.font == dataset.font_classes[0]
    assert sample.ref.codepoint == ord("A")
    assert sample.ref.location == {}
    assert sample.font_idx == 0
    assert sample.style_idx == 0
    assert sample.character_idx == 0


def test_load_glyph_returns_outline_tensors() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[ord("o")],
    )

    types, coords = load_glyph(dataset[0].ref)

    assert types.dtype == torch.long
    assert types.ndim == 1
    assert coords.dtype == torch.float32
    assert coords.ndim == 2
    assert coords.shape[1] == 6
    assert (types == ElementType.QUAD_TO.value).any().item()
    assert not (types == ElementType.CURVE_TO.value).any().item()


def test_glyph_dataset_transform_uses_sample_first_contract() -> None:
    calls: list[GlyphSample] = []

    def transform(sample: GlyphSample) -> tuple[torch.Tensor, int]:
        calls.append(sample)
        types, _coords = load_glyph(sample.ref)
        return types[:2], sample.character_idx

    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
        transform=transform,
    )

    types, character_idx = dataset[0]

    assert len(calls) == 1
    assert isinstance(calls[0], GlyphSample)
    assert types.shape[0] == 2
    assert character_idx == 0


def test_glyph_dataset_variable_fonts_named_instances() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
    )

    assert len(dataset) == len(dataset.style_classes)
    assert "Roboto wght=100,wdth=100" in dataset.style_classes
    assert "Roboto wght=400,wdth=75" in dataset.style_classes
    assert not hasattr(dataset, "weight_targets")
    assert not hasattr(dataset, "width_targets")
    assert not hasattr(dataset, "slant_targets")
    assert not hasattr(dataset, "italic_targets")
    assert not hasattr(dataset, "optical_size_targets")
    sample = dataset[0]
    assert not hasattr(sample, "weight_targets")
    assert not hasattr(sample, "width_targets")
    assert not hasattr(sample, "slant_targets")
    assert not hasattr(sample, "italic_targets")
    assert not hasattr(sample, "optical_size_targets")


def test_default_and_grid_instance_functions() -> None:
    default_dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=default_instance,
    )
    grid_dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=grid_instances({"wght": 2, "wdth": 2}),
    )

    assert len(default_dataset) == 1
    assert default_dataset[0].ref.location == {"wght": 400.0, "wdth": 100.0}
    assert len(grid_dataset) == 4
    assert [grid_dataset[i].ref.location for i in range(len(grid_dataset))] == [
        {"wght": 100.0, "wdth": 75.0},
        {"wght": 100.0, "wdth": 100.0},
        {"wght": 900.0, "wdth": 75.0},
        {"wght": 900.0, "wdth": 100.0},
    ]


def test_instance_fn_can_return_zero_locations() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
        instances=lambda _font: [],
    )

    assert len(dataset) == 0
    assert dataset.font_classes == []
    assert dataset.style_classes == []
    assert dataset.character_classes == []
    assert dataset.style_targets.shape == (0,)
    assert dataset.character_targets.shape == (0,)


def test_variable_glyph_dataset_instance_count_refs_without_styles() -> None:
    dataset = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41, 0x42],
        instance_count=lambda _font: 2,
    )

    assert repr(dataset) == (
        f"VariableGlyphDataset(root={str(dataset.root)!r}, samples=4, "
        "font_classes=1, character_classes=2)"
    )
    assert len(dataset) == 4
    assert dataset.font_targets.tolist() == [0, 0, 0, 0]
    assert dataset.character_targets.tolist() == [0, 1, 0, 1]

    sample = dataset[0]

    assert isinstance(sample, VariableGlyphSample)
    assert isinstance(sample.ref, VariableGlyphRef)
    assert sample.ref.codepoint == 0x41
    assert sample.font_idx == 0
    assert sample.character_idx == 0

    location = random_location(
        sample.ref.font, generator=torch.Generator().manual_seed(5)
    )
    types, coords = load_glyph(sample.ref, location)
    assert types.ndim == 1
    assert coords.shape[1] == 6


def test_variable_glyph_dataset_defaults_to_named_instance_count() -> None:
    fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41, 0x42],
        instances=named_instances,
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41, 0x42],
    )

    assert len(variable) == len(fixed)
    assert variable.character_targets.tolist() == fixed.character_targets.tolist()
    assert "instance_count" not in variable.__dict__


def test_instance_count_fns_match_instance_fn_multiplicity() -> None:
    named_fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=named_instances,
    )
    named_variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=named_instance_count,
    )
    default_variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=default_instance_count,
    )
    grid_variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=grid_instance_count({"wght": 2, "wdth": 2}),
    )

    assert len(named_variable) == len(named_fixed)
    assert len(default_variable) == 1
    assert len(grid_variable) == 4


def test_instance_count_fns_keep_static_fonts_at_one_slot() -> None:
    for instance_count in [
        default_instance_count,
        named_instance_count,
        grid_instance_count({"wght": 2}),
    ]:
        dataset = VariableGlyphDataset(
            root="tests/fonts",
            patterns=("lato/Lato-Regular.ttf",),
            codepoints=[0x41],
            instance_count=instance_count,
        )
        assert len(dataset) == 1


def test_variable_glyph_dataset_transform_can_sample_location() -> None:
    dataset = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=lambda _font: 1,
        transform=_variable_to_pair,
    )

    types, coords = dataset[0]

    assert types.ndim == 1
    assert coords.shape[1] == 6


@pytest.mark.parametrize("codepoint", [1.5, "A"])
def test_glyph_dataset_rejects_non_integer_codepoints(codepoint: object) -> None:
    with pytest.raises(TypeError, match="cannot be interpreted as an integer"):
        GlyphDataset(
            root="tests/fonts",
            patterns=("lato/Lato-Regular.ttf",),
            codepoints=[codepoint],  # ty: ignore[invalid-argument-type]
        )


def test_glyph_dataset_negative_indexing_and_bounds() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    assert dataset[-1] == dataset[len(dataset) - 1]
    with pytest.raises(IndexError):
        dataset[len(dataset)]
    with pytest.raises(IndexError):
        dataset[-len(dataset) - 1]


def test_pattern_filter_empty_result_and_outline_less_fonts() -> None:
    empty = GlyphDataset(
        root="tests/fonts",
        patterns=("nonexistent*.ttf",),
        codepoints=range(0x80),
    )
    no_outlines = GlyphDataset(
        root="tests/fonts",
        patterns=("nocolortest/NoOutlines-Regular.ttf",),
        codepoints=range(0x80),
    )

    assert len(empty) == 0
    assert empty.style_targets.shape == (0,)
    assert empty.character_targets.shape == (0,)
    assert len(no_outlines) == 0
    with pytest.raises(IndexError):
        no_outlines[0]


def test_glyph_dataset_discovers_fonts_in_hidden_directories(tmp_path: Path) -> None:
    source = Path("tests/fonts/lato/Lato-Regular.ttf").resolve()
    hidden_dir = tmp_path / ".fonts"
    hidden_dir.mkdir()
    shutil.copy(source, hidden_dir / "Lato-Regular.ttf")

    dataset = GlyphDataset(root=tmp_path, codepoints=[0x41])
    sample = dataset[0]

    assert len(dataset) == 1
    assert Path(sample.ref.font.path).name == "Lato-Regular.ttf"


def test_glyph_dataset_supports_non_utf8_font_paths(tmp_path: Path) -> None:
    if os.name == "nt":
        pytest.skip("Windows paths are Unicode")

    source = Path("tests/fonts/lato/Lato-Regular.ttf").resolve()
    font_path = tmp_path / os.fsdecode(b"Lato-\xff.ttf")
    shutil.copy(source, font_path)

    dataset = GlyphDataset(root=tmp_path, codepoints=[0x41])
    sample = dataset[0]
    types, coords = load_glyph(sample.ref)

    assert len(dataset) == 1
    assert "\udcff" in sample.ref.font.path
    assert types.numel() > 0
    assert coords.shape[1] == 6


def test_glyph_dataset_ignores_gitignore_for_root_discovery(tmp_path: Path) -> None:
    source = Path("tests/fonts/lato/Lato-Regular.ttf").resolve()
    git_executable = shutil.which("git")
    if git_executable is None:
        pytest.skip("git not installed")
    subprocess.run(  # noqa: S603
        [git_executable, "init", "-q"],
        cwd=tmp_path,
        check=True,
    )
    font_path = tmp_path / "Lato-Regular.ttf"
    shutil.copy(source, font_path)
    (tmp_path / ".gitignore").write_text("*.ttf\n", encoding="utf-8")

    dataset = GlyphDataset(root=tmp_path, codepoints=[0x41])
    sample = dataset[0]

    assert len(dataset) == 1
    assert Path(sample.ref.font.path).name == "Lato-Regular.ttf"


def test_targets_match_samples() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=range(0x41, 0x44),
        instances=grid_instances({"wght": 2}),
    )

    assert dataset.style_targets.shape == (len(dataset),)
    assert dataset.style_targets.dtype == torch.long
    assert dataset.character_targets.shape == (len(dataset),)
    assert dataset.character_targets.dtype == torch.long
    assert dataset.font_targets.shape == (len(dataset),)
    for i in range(len(dataset)):
        sample = dataset[i]
        assert dataset.style_targets[i].item() == sample.style_idx
        assert dataset.character_targets[i].item() == sample.character_idx
        assert dataset.font_targets[i].item() == sample.font_idx


def test_datasets_public_api_is_ref_centered() -> None:
    assert datasets_module.FontRef is FontRef
    assert datasets_module.GlyphDataset is GlyphDataset
    assert datasets_module.GlyphSample is GlyphSample
    assert datasets_module.VariableGlyphDataset is VariableGlyphDataset
    assert not hasattr(datasets_module, "load_glyph")
    assert not hasattr(datasets_module, "DefaultInstantiation")
    assert not hasattr(datasets_module, "GridInstantiation")


def test_transforms_module_exports_load_glyph() -> None:
    assert transforms_module.load_glyph is load_glyph


def test_package_root_stays_thin() -> None:
    assert torchfont.__all__ == []
    assert not hasattr(torchfont, "GlyphDataset")
    assert not hasattr(torchfont, "GlyphSample")


def test_native_dataset_helpers_are_not_public_dataset_api() -> None:
    assert hasattr(_torchfont, "FixedGlyphIndex")
    assert hasattr(_torchfont, "VariableGlyphIndex")
    assert hasattr(_torchfont, "load_glyph")
    assert not hasattr(_torchfont, "FontIndexBackend")
    assert not hasattr(_torchfont, "FontInfo")
    assert not hasattr(_torchfont, "FixedGlyphLocation")
    assert not hasattr(_torchfont, "GlyphOutlineItem")
    assert not hasattr(_torchfont, "GlyphDataset")
    assert not hasattr(_torchfont, "GlyphDatasetBackend")
    assert not hasattr(_torchfont, "DefaultInstantiation")
    assert not hasattr(_torchfont, "GridInstantiation")
    assert not hasattr(_torchfont, "VariableGlyphLocation")
    assert not hasattr(_torchfont, "canonicalize_locations_for_font")
    assert not hasattr(_torchfont, "glyph_font_targets")
    assert not hasattr(_torchfont, "variable_glyph_font_targets")


def test_variation_module_exports_instance_functions() -> None:
    assert variation_module.default_instance is default_instance
    assert variation_module.default_instance_count is default_instance_count
    assert variation_module.named_instances is named_instances
    assert variation_module.named_instance_count is named_instance_count
    assert variation_module.grid_instances is grid_instances
    assert variation_module.grid_instance_count is grid_instance_count
    assert variation_module.random_location is random_location
    assert not hasattr(variation_module, "InstancePolicy")
    assert not hasattr(variation_module, "RepeatPolicy")
    assert not hasattr(variation_module, "default_repeats")
    assert not hasattr(variation_module, "grid_repeats")
    assert not hasattr(variation_module, "named_repeats")
    assert not hasattr(variation_module, "random_instances")
    assert not hasattr(variation_module, "random_instance_count")
    assert not hasattr(variation_module, "DefaultInstantiation")


def test_location_validation_rejects_unknown_axis_range_and_nan() -> None:
    dataset = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=lambda _font: 1,
    )
    ref = dataset[0].ref

    with pytest.raises(ValueError, match="no variation axis 'xxxx'"):
        load_glyph(ref, {"xxxx": 1.0})
    with pytest.raises(ValueError, match="outside"):
        load_glyph(ref, {"wght": 10_000.0})
    with pytest.raises(ValueError, match="finite"):
        load_glyph(ref, {"wght": float("nan")})


def test_missing_instance_location_axes_use_defaults() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=lambda _font: [{"wght": 400.0}],
    )

    assert len(dataset) == 1
    assert dataset[0].ref.location == {"wght": 400.0, "wdth": 100.0}


def test_instance_fn_rejects_duplicate_normalized_locations() -> None:
    with pytest.raises(ValueError, match="duplicate variation locations"):
        GlyphDataset(
            root="tests/fonts",
            patterns=("roboto/Roboto*.ttf",),
            codepoints=[0x41],
            instances=lambda _font: [{"wght": 400.0}, {"wght": 400.0}],
        )


def test_instance_fn_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="no variation axis 'xxxx'"):
        GlyphDataset(
            root="tests/fonts",
            patterns=("roboto/Roboto*.ttf",),
            codepoints=[0x41],
            instances=lambda _font: [{"xxxx": 1.0}],
        )


@pytest.mark.parametrize("axes", [{}, {"wght": 0}, {"wght": -1}])
def test_grid_functions_reject_invalid_axis_counts(axes: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="grid_instances"):
        GlyphDataset(
            root="tests/fonts",
            patterns=("roboto/Roboto*.ttf",),
            codepoints=[0x41],
            instances=grid_instances(axes),
        )
    with pytest.raises(ValueError, match="grid_instances"):
        VariableGlyphDataset(
            root="tests/fonts",
            patterns=("roboto/Roboto*.ttf",),
            codepoints=[0x41],
            instance_count=grid_instance_count(axes),
        )


@pytest.mark.parametrize("axes", [{}, {"wght": 0}, {"wght": -1}])
def test_native_grid_locations_reject_invalid_axis_counts(
    axes: dict[str, int],
) -> None:
    with pytest.raises(ValueError, match="grid_instances"):
        _torchfont.grid_locations_for_font(
            "tests/fonts/roboto/Roboto[wdth,wght].ttf",
            0,
            axes,
        )


def test_grid_functions_ignore_unknown_axes_and_pin_unlisted_axes_to_default() -> None:
    fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=grid_instances({"wght": 2, "xxxx": 99}),
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=grid_instance_count({"wght": 2, "xxxx": 99}),
    )

    assert len(fixed) == 2
    assert len(variable) == 2
    assert [fixed[i].ref.location for i in range(len(fixed))] == [
        {"wght": 100.0, "wdth": 100.0},
        {"wght": 900.0, "wdth": 100.0},
    ]


def test_grid_functions_use_default_when_no_requested_axes_exist() -> None:
    fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=grid_instances({"xxxx": 2}),
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_count=grid_instance_count({"xxxx": 2}),
    )

    assert len(fixed) == 1
    assert len(variable) == 1
    assert fixed[0].ref.location == {"wght": 400.0, "wdth": 100.0}


def test_grid_instances_keeps_static_fonts_at_default() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[0x41],
        instances=grid_instances({"wght": 2}),
    )

    assert len(dataset) == 1
    assert dataset[0].ref.location == {}


def test_variation_survives_pickle_without_instance_fn() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instances=grid_instances({"wght": 2}),
    )

    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301

    assert [restored[i].ref.location for i in range(len(restored))] == [
        dataset[i].ref.location for i in range(len(dataset))
    ]
    assert "instances" not in restored.__dict__


@pytest.mark.parametrize("start_method", [None, *mp.get_all_start_methods()])
def test_glyph_dataset_dataloader_multiworker(start_method: str | None) -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x5B),
        transform=_to_pair,
    )

    loader = DataLoader(
        dataset,
        batch_size=1,
        num_workers=2,
        shuffle=False,
        multiprocessing_context=start_method,
        collate_fn=_collate_outline,
    )

    types_t, coords_t = next(iter(loader))

    assert types_t.dtype == torch.long
    assert types_t.ndim == 2
    assert coords_t.dtype == torch.float32
    assert coords_t.ndim == 3
    assert coords_t.shape[2] == 6


def test_target_vectors_survive_pickle() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
    )

    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301

    assert torch.equal(restored.style_targets, dataset.style_targets)
    assert torch.equal(restored.character_targets, dataset.character_targets)
    assert torch.equal(restored.font_targets, dataset.font_targets)


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
    font_idx, character_idx, types_len, coords_shape = queue.get(timeout=5)
    assert font_idx == 0
    assert character_idx == 0
    assert types_len > 0
    assert coords_shape[1] == 6
