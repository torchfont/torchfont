from collections.abc import Callable

import pytest
import torch

from torchfont.io import ElementType


@pytest.fixture
def simple_outline() -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.LINE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 1.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return types, coords


@pytest.fixture
def cubic_outline() -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.CURVE_TO.value,
            ElementType.CURVE_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.2, 0.8, 0.8, 0.8, 1.0, 0.0],
            [0.8, 0.2, 0.2, 0.2, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return types, coords


@pytest.fixture
def quad_outline() -> tuple[torch.Tensor, torch.Tensor]:
    types = torch.tensor(
        [
            ElementType.MOVE_TO.value,
            ElementType.QUAD_TO.value,
            ElementType.QUAD_TO.value,
            ElementType.CLOSE.value,
            ElementType.END.value,
        ],
        dtype=torch.long,
    )
    coords = torch.tensor(
        [
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.5, 1.0, 0.0, 0.0, 1.0, 0.0],
            [0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ],
        dtype=torch.float32,
    )
    return types, coords


@pytest.fixture
def square() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                ElementType.MOVE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.CLOSE.value,
                ElementType.END.value,
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [
                [0, 0, 0, 0, 1.0, 1.0],
                [0, 0, 0, 0, 2.0, 1.0],
                [0, 0, 0, 0, 2.0, 2.0],
                [0, 0, 0, 0, 1.0, 2.0],
                [0, 0, 0, 0, 0.0, 0.0],
                [0, 0, 0, 0, 0.0, 0.0],
            ],
            dtype=torch.float32,
        ),
    )


@pytest.fixture
def open_subpath() -> tuple[torch.Tensor, torch.Tensor]:
    return (
        torch.tensor(
            [
                ElementType.MOVE_TO.value,
                ElementType.LINE_TO.value,
                ElementType.END.value,
            ],
            dtype=torch.long,
        ),
        torch.tensor(
            [[0, 0, 0, 0, 2.0, 0.0], [0, 0, 0, 0, 1.0, 0.0], [0] * 6],
            dtype=torch.float32,
        ),
    )


@pytest.fixture
def close_end_zeros() -> Callable[[torch.Tensor, torch.Tensor], bool]:
    def check(types: torch.Tensor, coords: torch.Tensor) -> bool:
        for cmd in (ElementType.CLOSE.value, ElementType.END.value):
            idx = types.tolist().index(cmd)
            if not torch.all(coords[idx] == 0.0):
                return False
        return True

    return check
