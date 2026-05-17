import torch

from torchfont.io import ElementType


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
