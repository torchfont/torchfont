"""Shared neural network helpers for outline tensors."""

from __future__ import annotations

import torch
from torch import Tensor

from torchfont._outline import ElementType


def _active_coordinate_mask(types: Tensor) -> Tensor:
    pair0 = (types == ElementType.QUAD_TO.value) | (types == ElementType.CURVE_TO.value)
    pair1 = types == ElementType.CURVE_TO.value
    pair2 = (
        (types == ElementType.MOVE_TO.value)
        | (types == ElementType.LINE_TO.value)
        | (types == ElementType.QUAD_TO.value)
        | (types == ElementType.CURVE_TO.value)
    )
    return torch.stack((pair0, pair0, pair1, pair1, pair2, pair2), dim=-1)
