import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import RandomTruncateSubpaths, functional


@pytest.fixture
def three_lines() -> Outline:
    types = torch.tensor(
        [
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.MOVE_TO,
            ElementType.LINE_TO,
            ElementType.END,
        ],
        dtype=torch.long,
    )
    coords = torch.zeros(7, 6)
    coords[:, 4] = torch.arange(7)
    return Outline(types, coords)


def test_drop_subpaths_to_fit_uses_explicit_removal_order(
    three_lines: Outline,
) -> None:
    output = functional.drop_subpaths_to_fit(
        three_lines,
        torch.tensor([0.1, 0.0, 0.9, 0.0, 0.2, 0.0, 0.0]),
        max_subpaths=2,
    )

    assert output.coords[output.types == ElementType.MOVE_TO, 4].tolist() == [
        2.0,
        4.0,
    ]


def test_random_truncate_subpaths_meets_both_limits(three_lines: Outline) -> None:
    output = RandomTruncateSubpaths(max_length=3, max_subpaths=2)(three_lines)

    assert len(output) <= 3
    assert output.types.tolist().count(ElementType.MOVE_TO) <= 2


def test_random_truncate_subpaths_zero_count_returns_empty(
    three_lines: Outline,
) -> None:
    output = RandomTruncateSubpaths(max_subpaths=0)(three_lines)

    assert output.types.tolist() == [ElementType.END]


def test_random_truncate_subpaths_is_reproducible(three_lines: Outline) -> None:
    torch.manual_seed(0)
    first = RandomTruncateSubpaths(max_subpaths=1)(three_lines)
    torch.manual_seed(0)
    second = RandomTruncateSubpaths(max_subpaths=1)(three_lines)

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


def test_random_truncate_subpaths_shares_removal_order_between_outlines(
    three_lines: Outline,
) -> None:
    first, second = RandomTruncateSubpaths(max_subpaths=1)([three_lines, three_lines])

    assert torch.equal(first.types, second.types)
    assert torch.equal(first.coords, second.coords)


def test_random_truncate_subpaths_requires_a_limit() -> None:
    with pytest.raises(
        ValueError, match="max_length or max_subpaths must be specified"
    ):
        RandomTruncateSubpaths()


def test_random_truncate_subpaths_repr() -> None:
    assert (
        repr(RandomTruncateSubpaths(max_length=128, max_subpaths=16))
        == "RandomTruncateSubpaths(max_length=128, max_subpaths=16)"
    )
