from __future__ import annotations

import multiprocessing as mp
import os
import pickle
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Any, cast

import pytest
import torch
from fontTools.ttLib import TTFont
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

import torchfont.datasets as datasets_module
import torchfont.instance_fn as instance_fn_module
import torchfont.structures as structures_module
import torchfont.transforms as transforms_module
from torchfont import _torchfont
from torchfont.datasets import (
    GlyphDataset,
    VariableGlyphDataset,
)
from torchfont.instance_fn import (
    default_instance,
    default_instance_count,
    grid_instance_count,
    grid_instances,
    named_instance_count,
    named_instances,
)
from torchfont.structures import (
    ElementType,
    FontRef,
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
)
from torchfont.transforms import RandomLocation
from torchfont.transforms import functional as _functional

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping


class _WeightTag:
    def __str__(self) -> str:
        return "wght"


class _TestInstances:
    def __init__(
        self,
        locations: list[Mapping[str, float]] | None = None,
        *,
        count: int | None = None,
    ) -> None:
        self._locations = locations or []
        self._count = len(self._locations) if count is None else count

    def locations(self, font: FontRef) -> Iterable[Mapping[str, float]]:
        del font
        return self._locations

    def count(self, font: FontRef) -> int:
        del font
        return self._count


def _load_pair(
    ref: GlyphRef | VariableGlyphRef,
    location: dict[str, float] | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    outline = (
        _functional.load_glyph(ref)
        if isinstance(ref, GlyphRef)
        else _functional.load_glyph(ref, location)
    )
    return outline.types, outline.coords


def _to_pair(sample: GlyphSample) -> tuple[torch.Tensor, torch.Tensor]:
    return _load_pair(sample.ref)


def _variable_to_pair(
    sample: VariableGlyphSample,
) -> tuple[torch.Tensor, torch.Tensor]:
    data = RandomLocation()(sample)
    return data.data.types, data.data.coords


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
    types, coords = _load_pair(sample.ref)
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


@pytest.mark.parametrize("dataset_type", [GlyphDataset, VariableGlyphDataset])
def test_dataset_accepts_one_pattern_as_a_string(dataset_type: type) -> None:
    dataset = dataset_type(
        root="tests/fonts",
        patterns="lato/Lato-Regular.ttf",
        codepoints=[0x41],
    )

    assert dataset.patterns == ("lato/Lato-Regular.ttf",)
    assert len(dataset) == 1


def test_load_glyph_returns_outline_tensors() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[ord("o")],
    )

    types, coords = _load_pair(dataset[0].ref)

    assert types.dtype == torch.long
    assert types.ndim == 1
    assert coords.dtype == torch.float32
    assert coords.ndim == 2
    assert coords.shape[1] == 6
    assert (types == ElementType.QUAD_TO.value).any().item()
    assert not (types == ElementType.CURVE_TO.value).any().item()


def test_dataset_reports_corrupt_font_parse_error(tmp_path: Path) -> None:
    (tmp_path / "broken.ttf").write_bytes(b"not a font")

    with pytest.raises(ValueError, match="failed to parse"):
        GlyphDataset(root=tmp_path)


def test_dataset_reports_missing_root_as_file_not_found(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="failed to resolve font root"):
        GlyphDataset(root=tmp_path / "missing")


def test_load_glyph_reports_missing_font_as_file_not_found(tmp_path: Path) -> None:
    ref = GlyphRef(FontRef(str(tmp_path / "missing.ttf"), 0), ord("A"), {})

    with pytest.raises(FileNotFoundError, match="failed to open"):
        _load_pair(ref)


def test_dataset_reports_invalid_pattern_as_value_error() -> None:
    with pytest.raises(ValueError, match="invalid pattern"):
        GlyphDataset(root="tests/fonts", patterns=("[",))


def test_glyph_dataset_transform_uses_sample_first_contract() -> None:
    calls: list[GlyphSample] = []

    def transform(sample: GlyphSample) -> tuple[torch.Tensor, int]:
        calls.append(sample)
        types, _coords = _load_pair(sample.ref)
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


def test_default_and_grid_instance_functions() -> None:
    default_dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=default_instance,
    )
    grid_dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instances({"wght": 2, "wdth": 2}),
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


def test_instance_function_can_return_zero_locations() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=range(0x41, 0x44),
        instance_fn=_TestInstances().locations,
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
        instance_fn=_TestInstances(count=2).count,
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

    torch.manual_seed(5)
    data = RandomLocation()(sample)
    types, coords = data.data.types, data.data.coords
    assert types.ndim == 1
    assert coords.shape[1] == 6


def test_variable_glyph_dataset_defaults_to_named_instances() -> None:
    fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41, 0x42],
        instance_fn=named_instances,
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41, 0x42],
    )

    assert len(variable) == len(fixed)
    assert variable.character_targets.tolist() == fixed.character_targets.tolist()


def test_instance_functions_match_fixed_and_variable_multiplicity() -> None:
    named_fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=named_instances,
    )
    named_variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=named_instance_count,
    )
    default_variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=default_instance_count,
    )
    grid_variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instance_count({"wght": 2, "wdth": 2}),
    )

    assert len(named_variable) == len(named_fixed)
    assert len(default_variable) == 1
    assert len(grid_variable) == 4


def test_instance_count_functions_keep_static_fonts_at_one_slot() -> None:
    for instance_fn in [
        default_instance_count,
        named_instance_count,
        grid_instance_count({"wght": 2}),
    ]:
        dataset = VariableGlyphDataset(
            root="tests/fonts",
            patterns=("lato/Lato-Regular.ttf",),
            codepoints=[0x41],
            instance_fn=instance_fn,
        )
        assert len(dataset) == 1


def test_variable_glyph_dataset_transform_can_sample_location() -> None:
    dataset = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=_TestInstances(count=1).count,
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
    types, coords = _load_pair(sample.ref)

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
        instance_fn=grid_instances({"wght": 2}),
    )

    assert dataset.style_targets.shape == (len(dataset),)
    assert dataset.style_targets.dtype == torch.long
    assert dataset.character_targets.shape == (len(dataset),)
    assert dataset.character_targets.dtype == torch.long
    assert dataset.font_targets.shape == (len(dataset),)
    axis_targets = (
        dataset.weight_targets,
        dataset.width_targets,
        dataset.italic_targets,
        dataset.slant_targets,
        dataset.optical_size_targets,
    )
    for i in range(len(dataset)):
        sample = dataset[i]
        assert dataset.style_targets[i].item() == sample.style_idx
        assert dataset.character_targets[i].item() == sample.character_idx
        assert dataset.font_targets[i].item() == sample.font_idx
        for targets, value in zip(
            axis_targets,
            (
                sample.weight,
                sample.width,
                sample.italic,
                sample.slant,
                sample.optical_size,
            ),
            strict=True,
        ):
            expected = float("nan") if value is None else value
            assert torch.allclose(
                targets[i],
                torch.tensor(expected),
                equal_nan=True,
            )


def test_axis_targets_follow_locations_and_include_static_fonts() -> None:
    variable = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instances({"wght": 2, "wdth": 2}),
    )
    static = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[0x41],
    )

    for targets in (
        variable.weight_targets,
        variable.width_targets,
        variable.italic_targets,
        variable.slant_targets,
        variable.optical_size_targets,
    ):
        assert targets.shape == (4,)
        assert targets.dtype == torch.float32
    assert torch.equal(
        variable.weight_targets,
        torch.tensor([100.0, 100.0, 900.0, 900.0]),
    )
    assert torch.equal(
        variable.width_targets,
        torch.tensor([75.0, 100.0, 75.0, 100.0]),
    )
    assert static.weight_targets.tolist() == [400.0]
    assert static.width_targets.tolist() == [100.0]
    assert static.italic_targets.tolist() == [0.0]
    assert static.slant_targets.tolist() == [0.0]
    assert torch.isnan(static.optical_size_targets[0])


def test_head_bold_is_not_invented_as_a_weight_class(tmp_path: Path) -> None:
    font = TTFont("tests/fonts/lato/Lato-Regular.ttf")
    cast("Any", font["head"]).macStyle |= 0b11
    del font["OS/2"]
    font.save(tmp_path / "Lato-No-OS2.ttf")

    dataset = GlyphDataset(root=tmp_path, codepoints=[0x41])

    # head.macStyle records only a binary bold flag; it does not distinguish
    # 600, 700, 800, 900, or any other weight class.
    assert torch.isnan(dataset.weight_targets[0])
    assert dataset.italic_targets[0].item() == 1.0


def test_datasets_public_api_is_ref_centered() -> None:
    assert structures_module.FontRef is FontRef
    assert datasets_module.GlyphDataset is GlyphDataset
    assert structures_module.GlyphSample is GlyphSample
    assert datasets_module.VariableGlyphDataset is VariableGlyphDataset


def test_transforms_module_exports_class_based_dataset_transforms() -> None:
    assert transforms_module.LoadGlyph.__name__ == "LoadGlyph"
    assert transforms_module.RandomLocation.__name__ == "RandomLocation"


def test_native_dataset_helpers_are_available() -> None:
    assert hasattr(_torchfont, "FixedGlyphIndex")
    assert hasattr(_torchfont, "VariableGlyphIndex")
    assert hasattr(_torchfont, "load_glyph")


def test_instance_function_module_exports_builtins() -> None:
    assert instance_fn_module.default_instance is default_instance
    assert instance_fn_module.named_instances is named_instances
    assert instance_fn_module.grid_instances is grid_instances


def test_location_validation_rejects_unknown_axis_range_and_nan() -> None:
    dataset = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=_TestInstances(count=1).count,
    )
    ref = dataset[0].ref

    with pytest.raises(ValueError, match="no variation axis 'xxxx'"):
        _load_pair(ref, {"xxxx": 1.0})
    with pytest.raises(ValueError, match="outside"):
        _load_pair(ref, {"wght": 10_000.0})
    with pytest.raises(ValueError, match="finite"):
        _load_pair(ref, {"wght": float("nan")})
    with pytest.raises(ValueError, match="duplicate variation axis tag 'wght'"):
        _load_pair(
            ref,
            {"wght": 400.0, _WeightTag(): 500.0},  # ty: ignore[invalid-argument-type]
        )


def test_missing_instance_location_axes_use_defaults() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=_TestInstances([{"wght": 400.0}]).locations,
    )

    assert len(dataset) == 1
    assert dataset[0].ref.location == {"wght": 400.0, "wdth": 100.0}


def test_instance_function_rejects_duplicate_normalized_locations() -> None:
    with pytest.raises(ValueError, match="duplicate variation locations"):
        GlyphDataset(
            root="tests/fonts",
            patterns=("roboto/Roboto*.ttf",),
            codepoints=[0x41],
            instance_fn=_TestInstances([{"wght": 400.0}, {"wght": 400.0}]).locations,
        )


def test_instance_function_rejects_duplicate_normalized_axis_tags() -> None:
    location = cast(
        "Mapping[str, float]",
        {
            "wght": 400.0,
            _WeightTag(): 500.0,
        },
    )
    with pytest.raises(ValueError, match="duplicate variation axis tag 'wght'"):
        GlyphDataset(
            root="tests/fonts",
            patterns="roboto/Roboto*.ttf",
            codepoints=[0x41],
            instance_fn=_TestInstances([location]).locations,
        )


def test_instance_function_accepts_location_iterables() -> None:
    def locations(_font: FontRef) -> Iterable[Mapping[str, float]]:
        yield {"wght": 400.0}

    dataset = GlyphDataset(
        root="tests/fonts",
        patterns="roboto/Roboto*.ttf",
        codepoints=[0x41],
        instance_fn=locations,
    )

    assert len(dataset) == 1


def test_instance_function_rejects_unknown_axis() -> None:
    with pytest.raises(ValueError, match="no variation axis 'xxxx'"):
        GlyphDataset(
            root="tests/fonts",
            patterns=("roboto/Roboto*.ttf",),
            codepoints=[0x41],
            instance_fn=_TestInstances([{"xxxx": 1.0}]).locations,
        )


@pytest.mark.parametrize("axes", [{"wght": 0}, {"wght": -1}])
def test_grid_functions_reject_invalid_axis_counts(axes: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="ax"):
        grid_instances(axes)
    with pytest.raises(ValueError, match="ax"):
        grid_instance_count(axes)


def test_grid_functions_reject_duplicate_normalized_axis_tags() -> None:
    axes = {"wght": 2, _WeightTag(): 3}

    with pytest.raises(ValueError, match="duplicate variation axis tag 'wght'"):
        grid_instances(axes)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="duplicate variation axis tag 'wght'"):
        grid_instance_count(axes)  # ty: ignore[invalid-argument-type]


def test_empty_grid_functions_select_one_default_instance() -> None:
    fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instances({}),
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instance_count({}),
    )

    assert len(fixed) == 1
    assert len(variable) == 1
    assert fixed[0].ref.location == {"wght": 400.0, "wdth": 100.0}


def test_grid_functions_ignore_unknown_axes_and_pin_unlisted_axes() -> None:
    fixed = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instances({"wght": 2, "xxxx": 99}),
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instance_count({"wght": 2, "xxxx": 99}),
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
        instance_fn=grid_instances({"xxxx": 2}),
    )
    variable = VariableGlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instance_count({"xxxx": 2}),
    )

    assert len(fixed) == 1
    assert len(variable) == 1
    assert fixed[0].ref.location == {"wght": 400.0, "wdth": 100.0}


def test_grid_instances_keeps_static_fonts_at_default() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("lato/Lato-Regular.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instances({"wght": 2}),
    )

    assert len(dataset) == 1
    assert dataset[0].ref.location == {}


def test_variation_survives_pickle() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("roboto/Roboto*.ttf",),
        codepoints=[0x41],
        instance_fn=grid_instances({"wght": 2}),
    )

    restored = pickle.loads(pickle.dumps(dataset))  # noqa: S301

    assert [restored[i].ref.location for i in range(len(restored))] == [
        dataset[i].ref.location for i in range(len(dataset))
    ]


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
    for restored_targets, targets in (
        (restored.weight_targets, dataset.weight_targets),
        (restored.width_targets, dataset.width_targets),
        (restored.italic_targets, dataset.italic_targets),
        (restored.slant_targets, dataset.slant_targets),
        (restored.optical_size_targets, dataset.optical_size_targets),
    ):
        assert torch.allclose(restored_targets, targets, equal_nan=True)
    assert restored[0] == dataset[0]


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
