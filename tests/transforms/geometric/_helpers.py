import torch

from torchfont.io import ElementType


def _simple_outline() -> tuple[torch.Tensor, torch.Tensor]:
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


def _cubic_outline() -> tuple[torch.Tensor, torch.Tensor]:
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


def _quad_outline() -> tuple[torch.Tensor, torch.Tensor]:
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


def _close_end_zeros(types: torch.Tensor, coords: torch.Tensor) -> bool:
    for cmd in (ElementType.CLOSE.value, ElementType.END.value):
        idx = types.tolist().index(cmd)
        if not torch.all(coords[idx] == 0.0):
            return False
    return True
