import pickle
from typing import cast

import pytest
import torch
from typing_extensions import Self

from torchfont import tf_tensors
from torchfont.structures import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
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


def test_compose_materializes_module_iterables() -> None:
    child = AddToCoords(1.0)
    transform = Compose(iter([child]))

    assert transform.get_submodule("transforms.0") is child


def test_compose_rejects_plain_callables() -> None:
    with pytest.raises(TypeError, match="is not a Module subclass"):
        Compose([lambda value: value + 1])  # ty: ignore[invalid-argument-type]


def test_empty_compose_is_identity() -> None:
    outline = _line_outline()

    assert Compose([])(outline) is outline
    assert Compose([])(outline, "label") == (outline, "label")


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


def test_same_params_calls_wrapped_module_hooks() -> None:
    transform = AddToCoords(1.0)
    calls: list[object] = []
    transform.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output)
    )

    SameParams(transform)([_line_outline(), _line_outline()])

    assert len(calls) == 1


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
    )
    return GlyphSample(ref, 0, 0)


def test_load_and_outline_transforms_preserve_glyph_metadata() -> None:
    sample = _glyph_sample()
    transform = Compose([LoadGlyph(), QuadToCubic(merge_curves=True)])

    output = transform(sample)

    assert isinstance(output, GlyphData)
    assert isinstance(output.data, Outline)
    assert output.sample is sample
    assert output.location == {}


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


@pytest.mark.parametrize("shape", [(), (3,)])
def test_bitmap_rejects_data_with_fewer_than_two_dimensions(
    shape: tuple[int, ...],
) -> None:
    with pytest.raises(ValueError, match="Bitmap data must be at least 2-D"):
        tf_tensors.Bitmap(torch.zeros(shape))


def test_base_tf_tensor_wrap_rejects_unknown_metadata() -> None:
    bitmap = tf_tensors.Bitmap(torch.zeros((2, 2)))

    with pytest.raises(
        TypeError, match=r"Bitmap\.wrap\(\) does not accept metadata: label"
    ):
        tf_tensors.wrap(bitmap + 1, like=bitmap, label="ignored")


def test_custom_tf_tensor_controls_metadata_wrapping() -> None:
    class LabeledBitmap(tf_tensors.Bitmap):
        label: str

        def __new__(cls, data: object, *, label: str) -> Self:
            output = cast("Self", super().__new__(cls, data))
            output.label = label
            return output

        def __init__(self, data: object, *, label: str) -> None:
            del data, label

        @classmethod
        def wrap(
            cls,
            tensor: torch.Tensor,
            *,
            like: Self,
            **metadata: object,
        ) -> Self:
            output = tensor.as_subclass(cls)
            output.label = str(metadata.get("label", like.label))
            return output

    bitmap = LabeledBitmap(torch.zeros((2, 2)), label="source")

    cloned = bitmap.clone()
    relabeled = tf_tensors.wrap(bitmap + 1, like=bitmap, label="derived")

    assert isinstance(cloned, LabeledBitmap)
    assert cloned.label == "source"
    assert isinstance(relabeled, LabeledBitmap)
    assert relabeled.label == "derived"


def test_to_pure_tensor_removes_semantic_subclasses_in_pytrees() -> None:
    bitmap = tf_tensors.Bitmap(torch.zeros((2, 2)))

    output = ToPureTensor()({"bitmap": bitmap, "label": torch.tensor(1)})

    assert type(output["bitmap"]) is torch.Tensor
    assert output["bitmap"].data_ptr() == bitmap.data_ptr()
    assert type(output["label"]) is torch.Tensor


def test_random_location_returns_the_sampled_location() -> None:
    ref = GlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )
    sample = GlyphSample(ref, 0, 0)

    output = RandomLocation()(sample)

    assert isinstance(output, GlyphData)
    assert isinstance(output.data, Outline)
    assert output.sample is sample
    assert set(output.location) == {"wdth", "wght"}


def test_random_location_handles_multiple_variable_glyphs_independently() -> None:
    ref = GlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )

    first, second = RandomLocation()([ref, ref])

    assert isinstance(first, Outline)
    assert isinstance(second, Outline)


def test_random_location_can_share_parameters_within_one_font() -> None:
    font = FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0)
    first_sample = GlyphSample(GlyphRef(font, ord("A")), 0, 0)
    second_sample = GlyphSample(GlyphRef(font, ord("B")), 0, 1)

    first, second = SameParams(RandomLocation())([first_sample, second_sample])

    assert isinstance(first, GlyphData)
    assert isinstance(second, GlyphData)
    assert first.location == second.location


def test_random_location_rejects_shared_parameters_across_fonts() -> None:
    first = GlyphRef(
        FontRef("tests/fonts/roboto/Roboto[wdth,wght].ttf", 0),
        ord("A"),
    )
    second = GlyphRef(FontRef("tests/fonts/lato/Lato-Regular.ttf", 0), ord("A"))

    with pytest.raises(ValueError, match="only within one font"):
        SameParams(RandomLocation())([first, second])
