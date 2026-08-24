import pickle

import pytest
import torch

from tests._glyphs import glyph_id
from torchfont import (
    CodepointData,
    CodepointSample,
    ElementType,
    FontRef,
    GlyphRef,
    Outline,
)
from torchfont.transforms import (
    Compose,
    LoadGlyph,
    QuadToCubic,
    RandomApply,
    RandomSplitSegments,
    RenderBitmap,
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


def test_transform_shares_randomness_between_outlines() -> None:
    transform = RandomSplitSegments(split_probability=0.5)
    outline = _line_outline(8)

    first, second = transform([outline, outline])

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


def test_transform_calls_module_hooks_once() -> None:
    transform = AddToCoords(1.0)
    calls: list[object] = []
    transform.register_forward_hook(
        lambda _module, _inputs, output: calls.append(output)
    )

    transform([_line_outline(), _line_outline()])

    assert len(calls) == 1


def test_transform_preserves_module_pre_hook_inputs() -> None:
    transform = AddToCoords(1.0)
    outlines = [_line_outline(), _line_outline()]
    calls: list[tuple[object, ...]] = []
    transform.register_forward_pre_hook(lambda _module, inputs: calls.append(inputs))

    transform(outlines)

    assert len(calls) == 1
    assert calls[0][0] is outlines


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


def test_load_glyph_repr_contains_location_policy() -> None:
    assert repr(LoadGlyph()) == "LoadGlyph(location=default)"
    assert repr(LoadGlyph(location="random")) == "LoadGlyph(location=random)"


def test_load_glyph_rejects_invalid_location_policy() -> None:
    with pytest.raises(ValueError, match="location must be 'default' or 'random'"):
        LoadGlyph(location="invalid")  # ty: ignore[invalid-argument-type]


def _glyph_sample() -> CodepointSample:
    path = "tests/fonts/source-sans/SourceSans3-Regular.ttf"
    ref = GlyphRef(FontRef(path, 0), glyph_id(path, "A"))
    return CodepointSample(ref, ord("A"), 0, 0)


def test_load_and_outline_transforms_preserve_glyph_metadata() -> None:
    sample = _glyph_sample()
    transform = Compose([LoadGlyph(), QuadToCubic(merge_curves=True)])

    output = transform(sample)

    assert isinstance(output, CodepointData)
    assert isinstance(output.data, Outline)
    assert output.ref is sample.ref
    assert output.codepoint == sample.codepoint
    assert output.font_idx == sample.font_idx
    assert output.character_idx == sample.character_idx
    assert output.weight == 400.0
    assert output.width == 100.0
    assert output.italic == 0.0
    assert output.slant == 0.0
    assert output.optical_size is None
    assert output.location == {}


def test_type_changing_transforms_preserve_generic_glyph_container() -> None:
    sample = _glyph_sample()
    bitmap_output = Compose([LoadGlyph(), RenderBitmap(32)])(sample)

    assert isinstance(bitmap_output, CodepointData)
    assert type(bitmap_output.data) is torch.Tensor
    assert bitmap_output.data.shape == (32, 32)
    assert bitmap_output.ref is sample.ref
    assert bitmap_output.weight == 400.0


def test_load_glyph_returns_the_randomly_sampled_location() -> None:
    path = "tests/fonts/source-serif/SourceSerif4Variable-Roman.ttf"
    ref = GlyphRef(FontRef(path, 0), glyph_id(path, "A"))
    sample = CodepointSample(ref, ord("A"), 0, 0)

    output = LoadGlyph(location="random")(sample)

    assert isinstance(output, CodepointData)
    assert isinstance(output.data, Outline)
    assert output.ref is sample.ref
    assert output.weight == output.location["wght"]
    assert output.width == 100.0
    assert output.italic == 0.0
    assert output.slant == 0.0
    assert output.optical_size == output.location["opsz"]
    assert set(output.location) == {"opsz", "wght"}
