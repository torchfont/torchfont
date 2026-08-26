import pytest
import torch

from torchfont import ElementType, Outline
from torchfont.transforms import TruncateSubpaths, functional


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
    coords[:, 4] = torch.arange(5)
    return Outline(types, coords)


def test_truncate_subpaths_keeps_longest_fitting_prefix(two_lines: Outline) -> None:
    output = functional.truncate_subpaths(two_lines, 4)

    assert output.types.tolist() == [
        ElementType.MOVE_TO,
        ElementType.LINE_TO,
        ElementType.END,
    ]
    assert output.coords[:2, 4].tolist() == [0.0, 1.0]


def test_truncate_subpaths_does_not_partially_keep_first_subpath(
    two_lines: Outline,
) -> None:
    output = TruncateSubpaths(2)(two_lines)

    assert output.types.tolist() == [ElementType.END]
    assert output.coords.shape == (1, 6)


def test_truncate_subpaths_counts_end_in_limit(two_lines: Outline) -> None:
    output = TruncateSubpaths(5)(two_lines)

    assert torch.equal(output.types, two_lines.types)
    assert torch.equal(output.coords[:-1], two_lines.coords[:-1])


def test_truncate_subpaths_accepts_index_protocol(two_lines: Outline) -> None:
    output = TruncateSubpaths(torch.tensor(3), max_subpaths=torch.tensor(1))(two_lines)

    assert len(output) == 3


def test_truncate_subpaths_limits_subpath_count(two_lines: Outline) -> None:
    output = TruncateSubpaths(max_subpaths=1)(two_lines)

    assert output.types.tolist() == [
        ElementType.MOVE_TO,
        ElementType.LINE_TO,
        ElementType.END,
    ]


def test_truncate_subpaths_applies_both_limits(two_lines: Outline) -> None:
    output = TruncateSubpaths(max_length=5, max_subpaths=1)(two_lines)

    assert len(output) == 3


def test_zero_max_subpaths_returns_empty_outline(two_lines: Outline) -> None:
    output = functional.truncate_subpaths(two_lines, max_subpaths=0)

    assert output.types.tolist() == [ElementType.END]


@pytest.mark.parametrize("max_length", [0, -1])
def test_truncate_subpaths_rejects_non_positive_length(max_length: int) -> None:
    with pytest.raises(ValueError, match="max_length must be positive"):
        TruncateSubpaths(max_length)


def test_functional_truncate_subpaths_rejects_non_positive_length(
    two_lines: Outline,
) -> None:
    with pytest.raises(ValueError, match="max_length must be positive"):
        functional.truncate_subpaths(two_lines, 0)


def test_truncate_subpaths_requires_a_limit() -> None:
    with pytest.raises(
        ValueError, match="max_length or max_subpaths must be specified"
    ):
        TruncateSubpaths()


def test_functional_truncate_subpaths_requires_a_limit(two_lines: Outline) -> None:
    with pytest.raises(
        ValueError, match="max_length or max_subpaths must be specified"
    ):
        functional.truncate_subpaths(two_lines)


def test_truncate_subpaths_rejects_negative_subpath_count() -> None:
    with pytest.raises(ValueError, match="max_subpaths must be non-negative"):
        TruncateSubpaths(max_subpaths=-1)


def test_truncate_subpaths_repr() -> None:
    assert repr(TruncateSubpaths(128)) == "TruncateSubpaths(max_length=128)"
    assert (
        repr(TruncateSubpaths(max_subpaths=16)) == "TruncateSubpaths(max_subpaths=16)"
    )
