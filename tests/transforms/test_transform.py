import pickle

import pytest
import torch

from torchfont.datasets import (
    FontRef,
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
)
from torchfont.io import ElementType
from torchfont.transforms import (
    Bitmap,
    Compose,
    GlyphData,
    HorizontalFlip,
    LoadGlyph,
    Outline,
    QuadToCubic,
    RandomApply,
    RandomLocation,
    RandomSplitSegments,
    RenderBitmap,
    TFTensor,
    Transform,
)
from torchfont.transforms import functional as _functional


class AddToCoords(Transform):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def transform(self, outline: Outline, _params: dict[str, object]) -> Outline:
        return Outline(outline.types, outline.coords + self.value)


class IncrementInts(Transform):
    _transformed_types = (lambda value: isinstance(value, int),)

    def transform(self, inpt: object, _params: dict[str, object]) -> object:
        assert isinstance(inpt, int)
        return inpt + 1


def _line_outline(segment_count: int = 1) -> Outline:
    types = torch.tensor(
        [ElementType.MOVE_TO.value]
        + [ElementType.LINE_TO.value] * segment_count
        + [ElementType.END.value]
    )
    coords = torch.zeros((types.numel(), 6), dtype=torch.float32)
    coords[1 : segment_count + 1, 4] = torch.arange(1, segment_count + 1)
    return Outline(types, coords)


def test_transform_preserves_nested_structure_and_non_outline_leaves() -> None:
    first = _line_outline()
    second = _line_outline()
    inpt = {"outlines": [first, second], "label": 3}

    output = AddToCoords(2.0)(inpt)

    assert isinstance(output, dict)
    assert output["label"] == 3
    assert torch.equal(output["outlines"][0].coords, first.coords + 2.0)
    assert torch.equal(output["outlines"][1].coords, second.coords + 2.0)


def test_transform_type_predicates_select_semantic_leaves() -> None:
    assert IncrementInts()({"value": 1, "text": "1"}) == {
        "value": 2,
        "text": "1",
    }


def test_transform_evaluates_type_predicates_once_per_leaf() -> None:
    calls = 0

    def select_int(value: object) -> bool:
        nonlocal calls
        calls += 1
        return isinstance(value, int)

    class CountedSelection(IncrementInts):
        _transformed_types = (select_int,)

    assert CountedSelection()([1, "one"]) == [2, "one"]
    assert calls == 2


def test_compose_is_an_nn_module_and_applies_transforms_in_order() -> None:
    transform = Compose([AddToCoords(1.0), AddToCoords(2.0)])
    outline = _line_outline()

    output = transform(outline)

    assert isinstance(transform, torch.nn.Module)
    assert isinstance(transform, Transform)
    assert torch.equal(output.coords, outline.coords + 3.0)


def test_transform_and_compose_support_multiple_inputs() -> None:
    outline = _line_outline()

    output = Compose([AddToCoords(2.0)])(outline, "label")

    assert isinstance(output, tuple)
    assert torch.equal(output[0].coords, outline.coords + 2.0)
    assert output[1] == "label"


def test_compose_accepts_plain_callables() -> None:
    transform = Compose([lambda value: value + 1])

    assert transform(2) == 3


@pytest.mark.parametrize("container", [Compose, RandomApply])
def test_transform_containers_reject_empty_sequences(
    container: type[Compose] | type[RandomApply],
) -> None:
    with pytest.raises(ValueError, match="transforms must not be empty"):
        container([])


@pytest.mark.parametrize(("p", "expected"), [(0.0, 0.0), (1.0, 3.0)])
def test_random_apply_probability_boundaries(p: float, expected: float) -> None:
    outline = _line_outline()
    transform = RandomApply([AddToCoords(3.0)], p=p)

    output = transform(outline)

    assert isinstance(transform, Transform)
    assert torch.equal(output.coords, outline.coords + expected)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_random_apply_rejects_invalid_probability(p: float) -> None:
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        RandomApply([AddToCoords(1.0)], p=p)


def test_random_split_segments_handles_different_length_outlines() -> None:
    transform = RandomSplitSegments(split_probability=1.0, split_range=(0.5, 0.5))
    short = _line_outline(1)
    long = _line_outline(3)

    output = transform([short, long])

    assert output[0].types.numel() == short.types.numel() + 1
    assert output[1].types.numel() == long.types.numel() + 3


def test_random_split_segments_shares_randomness_between_outlines() -> None:
    transform = RandomSplitSegments(split_probability=0.5)
    outline = _line_outline(8)

    first, second = transform([outline, outline])

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


def test_transform_pipeline_is_pickleable() -> None:
    transform = Compose(
        [RandomApply([RandomSplitSegments(split_probability=1.0)], p=0.5)]
    )

    restored = pickle.loads(pickle.dumps(transform))  # noqa: S301

    assert isinstance(restored, Compose)
    assert restored.transforms[0].p == 0.5


def test_transform_repr_contains_configuration() -> None:
    transform = Compose(
        [RandomApply([RandomSplitSegments(split_probability=0.2)], p=0.3)]
    )

    representation = repr(transform)

    assert "p=0.3" in representation
    assert "split_probability=0.2" in representation


def _glyph_sample() -> GlyphSample:
    ref = GlyphRef(
        FontRef("tests/fonts/lato/Lato-Regular.ttf", 0),
        ord("A"),
        {},
    )
    return GlyphSample(ref, 0, 0, 0, None, None, None, None, None)


def test_load_and_outline_transforms_preserve_glyph_metadata() -> None:
    sample = _glyph_sample()
    transform = Compose([LoadGlyph(), QuadToCubic(merge_curves=True)])

    output = transform(sample)

    assert isinstance(output, GlyphData)
    assert isinstance(output.data, Outline)
    assert output.sample is sample
    assert output.location == sample.ref.location


def test_glyph_metadata_is_not_reprocessed_as_pytree_data() -> None:
    sample = _glyph_sample()
    first = LoadGlyph()(sample)

    second = LoadGlyph()(first)

    assert isinstance(second, GlyphData)
    assert second.sample is sample
    assert second.location is first.location
    assert second.data is first.data


def test_type_changing_transforms_preserve_generic_glyph_container() -> None:
    sample = _glyph_sample()
    bitmap_output = Compose([LoadGlyph(), RenderBitmap(32)])(sample)

    assert isinstance(bitmap_output, GlyphData)
    assert isinstance(bitmap_output.data, Bitmap)
    assert bitmap_output.data.shape == (32, 32)
    assert bitmap_output.sample is sample


def test_bitmap_behaves_as_a_tensor_and_preserves_its_type() -> None:
    bitmap = Bitmap(torch.zeros((8, 8), dtype=torch.uint8))

    output = pickle.loads(pickle.dumps(bitmap))  # noqa: S301

    assert isinstance(bitmap, TFTensor)
    assert type(bitmap + 1) is torch.Tensor
    assert isinstance(bitmap.to(dtype=torch.float32), Bitmap)
    assert isinstance(output, Bitmap)
    assert output.shape == (8, 8)


def test_random_location_returns_the_sampled_location() -> None:
    ref = VariableGlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )
    sample = VariableGlyphSample(ref, 0, 0)

    output = RandomLocation()(sample)

    assert isinstance(output, GlyphData)
    assert isinstance(output.data, Outline)
    assert output.sample is sample
    assert set(output.location) == {"wdth", "wght"}


def test_random_location_rejects_multiple_variable_glyphs() -> None:
    ref = VariableGlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )

    with pytest.raises(ValueError, match="requires exactly one"):
        RandomLocation()([ref, ref])


class CustomOutline(Outline):
    pass


@_functional.register_kernel(_functional.horizontal_flip, CustomOutline)
def _horizontal_flip_custom(
    inpt: CustomOutline, *, preserve_winding: bool = True
) -> Outline:
    del preserve_winding
    return Outline(inpt.types, inpt.coords + 10.0)


def test_functional_kernel_dispatch_supports_outline_subclasses() -> None:
    outline = _line_outline()
    custom = CustomOutline(outline.types, outline.coords)

    output = HorizontalFlip()(custom)

    assert torch.equal(output.coords, outline.coords + 10.0)
