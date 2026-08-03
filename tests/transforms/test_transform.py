import pickle

import pytest
import torch

from torchfont import tf_tensors
from torchfont.structures import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
    VariableGlyphRef,
    VariableGlyphSample,
)
from torchfont.transforms import (
    Compose,
    LoadGlyph,
    QuadToCubic,
    RandomApply,
    RandomLocation,
    RandomSplitSegments,
    RenderBitmap,
    SameParams,
    ToPureTensor,
    Transform,
)


class AddToCoords(Transform):
    def __init__(self, value: float) -> None:
        super().__init__()
        self.value = value

    def transform(self, outline: Outline, _params: dict[str, object]) -> Outline:
        return Outline(outline.types, outline.coords + self.value)


class IncrementInts(Transform):
    _transformed_types = (int,)

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


def test_transform_types_select_semantic_leaves() -> None:
    assert IncrementInts()({"value": 1, "text": "1"}) == {
        "value": 2,
        "text": "1",
    }


def test_compose_is_an_nn_module_and_applies_transforms_in_order() -> None:
    transform = Compose([AddToCoords(1.0), AddToCoords(2.0)])
    outline = _line_outline()

    output = transform(outline)

    assert isinstance(transform, torch.nn.Module)
    assert torch.equal(output.coords, outline.coords + 3.0)


def test_transform_and_compose_support_multiple_inputs() -> None:
    outline = _line_outline()

    output = Compose([AddToCoords(2.0)])(outline, "label")

    assert isinstance(output, tuple)
    assert torch.equal(output[0].coords, outline.coords + 2.0)
    assert output[1] == "label"


@pytest.mark.parametrize("as_module_list", [False, True])
def test_compose_registers_stateful_transforms(*, as_module_list: bool) -> None:
    class StatefulTransform(AddToCoords):
        def __init__(self) -> None:
            super().__init__(1.0)
            self.register_buffer("offset", torch.tensor(1.0))

    child = StatefulTransform()
    children = torch.nn.ModuleList([child]) if as_module_list else [child]
    transform = Compose(children)

    assert transform.get_submodule("transforms.0") is child
    assert transform.state_dict()["transforms.0.offset"].item() == 1.0


def test_compose_rejects_non_sequence_iterables() -> None:
    with pytest.raises(TypeError, match=r"sequence of nn\.Module"):
        Compose(iter([AddToCoords(1.0)]))  # ty: ignore[invalid-argument-type]


def test_compose_rejects_plain_callables() -> None:
    with pytest.raises(TypeError, match=r"only nn\.Module"):
        Compose([lambda value: value + 1])  # ty: ignore[invalid-argument-type]


def test_compose_rejects_empty_sequences() -> None:
    with pytest.raises(ValueError, match="transforms must not be empty"):
        Compose([])


@pytest.mark.parametrize(("p", "expected"), [(0.0, 0.0), (1.0, 3.0)])
def test_random_apply_probability_boundaries(p: float, expected: float) -> None:
    outline = _line_outline()
    transform = RandomApply(AddToCoords(3.0), p=p)

    output = transform(outline)

    assert torch.equal(output.coords, outline.coords + expected)


def test_random_apply_registers_transform() -> None:
    child = AddToCoords(1.0)
    transform = RandomApply(child, p=1.0)

    assert torch.equal(transform(_line_outline()).coords, _line_outline().coords + 1.0)
    assert transform.get_submodule("transform") is child


def test_random_apply_rejects_plain_callables() -> None:
    with pytest.raises(TypeError, match=r"must be an nn\.Module"):
        RandomApply(lambda value: value)  # ty: ignore[invalid-argument-type]


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_random_apply_rejects_invalid_probability(p: float) -> None:
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        RandomApply(AddToCoords(1.0), p=p)


def test_random_split_segments_handles_different_length_outlines() -> None:
    transform = RandomSplitSegments(split_probability=1.0, split_range=(0.5, 0.5))
    short = _line_outline(1)
    long = _line_outline(3)

    output = transform([short, long])

    assert output[0].types.numel() == short.types.numel() + 1
    assert output[1].types.numel() == long.types.numel() + 3


def test_same_params_shares_randomness_between_outlines() -> None:
    transform = RandomSplitSegments(split_probability=0.5)
    outline = _line_outline(8)

    first, second = SameParams(transform)([outline, outline])

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


def test_transform_pipeline_is_pickleable() -> None:
    transform = Compose(
        [RandomApply(RandomSplitSegments(split_probability=1.0), p=0.5)]
    )

    restored = pickle.loads(pickle.dumps(transform))  # noqa: S301

    assert isinstance(restored, Compose)
    assert restored.transforms[0].p == 0.5


def test_transform_repr_contains_configuration() -> None:
    transform = Compose(
        [RandomApply(RandomSplitSegments(split_probability=0.2), p=0.3)]
    )

    representation = repr(transform)

    assert "split_probability=0.2" in representation
    assert "p=0.3" in representation
    assert representation.count("RandomSplitSegments") == 1


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
    assert isinstance(bitmap_output.data, tf_tensors.Bitmap)
    assert bitmap_output.data.shape == (32, 32)
    assert bitmap_output.sample is sample


def test_bitmap_behaves_as_a_tensor_and_preserves_its_type() -> None:
    tensor = torch.zeros((8, 8), dtype=torch.uint8)
    bitmap = tf_tensors.Bitmap(tensor)

    output = pickle.loads(pickle.dumps(bitmap))  # noqa: S301

    assert isinstance(bitmap, tf_tensors.TFTensor)
    assert bitmap.data_ptr() == tensor.data_ptr()
    assert type(bitmap + 1) is torch.Tensor
    assert isinstance(bitmap.to(dtype=torch.float32), tf_tensors.Bitmap)
    assert type(bitmap.float()) is torch.Tensor
    assert type(bitmap.cpu()) is torch.Tensor
    assert isinstance(output, tf_tensors.Bitmap)
    assert output.shape == (8, 8)


def test_bitmap_accepts_tensor_like_data_and_can_be_rewrapped() -> None:
    bitmap = tf_tensors.Bitmap([[0, 1], [2, 3]], dtype=torch.float32)
    plain = bitmap + 1

    output = tf_tensors.wrap(plain, like=bitmap)

    assert bitmap.dtype is torch.float32
    assert isinstance(output, tf_tensors.Bitmap)
    assert torch.equal(output, torch.tensor([[1.0, 2.0], [3.0, 4.0]]))


def test_to_pure_tensor_removes_semantic_subclasses_in_pytrees() -> None:
    bitmap = tf_tensors.Bitmap(torch.zeros((2, 2)))

    output = ToPureTensor()({"bitmap": bitmap, "label": torch.tensor(1)})

    assert type(output["bitmap"]) is torch.Tensor
    assert output["bitmap"].data_ptr() == bitmap.data_ptr()
    assert type(output["label"]) is torch.Tensor


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


def test_random_location_handles_multiple_variable_glyphs_independently() -> None:
    ref = VariableGlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )

    first, second = RandomLocation()([ref, ref])

    assert isinstance(first, Outline)
    assert isinstance(second, Outline)


def test_random_location_rejects_shared_parameters() -> None:
    ref = VariableGlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )

    with pytest.raises(ValueError, match="cannot share parameters"):
        SameParams(RandomLocation())([ref, ref])
