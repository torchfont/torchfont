import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import RandomSubpathDropout, functional


@pytest.fixture
def two_lines() -> Outline:
    types = torch.tensor(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.END,
        ],
        dtype=torch.long,
    )
    coords = torch.zeros(5, 6)
    coords[0, 4:6] = torch.tensor([0.0, 0.0])
    coords[1, 4:6] = torch.tensor([1.0, 0.0])
    coords[2, 4:6] = torch.tensor([2.0, 0.0])
    coords[3, 4:6] = torch.tensor([3.0, 0.0])
    return Outline(types, coords)


def test_drop_subpaths_uses_explicit_selection_values(two_lines: Outline) -> None:
    output = functional.drop_subpaths(
        two_lines,
        torch.tensor([True, False]),
    )

    assert output.types.tolist() == [
        ElementType.MOVE_TO,
        ElementType.LINE_TO,
        ElementType.END,
    ]
    assert output.coords[0, 4].item() == 2.0


@pytest.mark.parametrize(("p", "subpath_count"), [(0.0, 2), (1.0, 0)])
def test_random_subpath_dropout_probability_boundaries(
    two_lines: Outline, p: float, subpath_count: int
) -> None:
    output = RandomSubpathDropout(p)(two_lines)

    assert output.types.tolist().count(ElementType.MOVE_TO) == subpath_count
    assert output.types[-1].item() == ElementType.END


def test_random_subpath_dropout_is_reproducible(two_lines: Outline) -> None:
    torch.manual_seed(0)
    first = RandomSubpathDropout(0.5)(two_lines)
    torch.manual_seed(0)
    second = RandomSubpathDropout(0.5)(two_lines)

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


def test_random_subpath_dropout_shares_selection_between_outlines(
    two_lines: Outline,
) -> None:
    first, second = RandomSubpathDropout(0.5)([two_lines, two_lines])

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


@pytest.mark.parametrize("p", [-0.1, 1.1])
def test_random_subpath_dropout_rejects_invalid_probability(p: float) -> None:
    with pytest.raises(ValueError, match="p must be between 0 and 1"):
        RandomSubpathDropout(p)


def test_random_subpath_dropout_uses_pytorch_default() -> None:
    assert repr(RandomSubpathDropout()) == "RandomSubpathDropout(p=0.5)"


def test_drop_subpaths_rejects_too_short_mask(two_lines: Outline) -> None:
    with pytest.raises(ValueError, match="at least the number of subpaths"):
        functional.drop_subpaths(
            two_lines,
            torch.tensor([False]),
        )


def test_drop_subpaths_requires_boolean_mask(two_lines: Outline) -> None:
    with pytest.raises(TypeError, match=r"mask must have dtype torch\.bool"):
        functional.drop_subpaths(two_lines, torch.tensor([0, 1]))
