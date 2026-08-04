"""Loss functions for outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional as _functional

from torchfont._outline import COORD_DIM, TYPE_DIM, ElementType
from torchfont.nn._utils import _active_coordinate_mask

if TYPE_CHECKING:
    from torch import Tensor

_Reduction = Literal["none", "mean", "sum"]


def coordinate_loss(
    prediction: Tensor,
    target: Tensor,
    types: Tensor,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Compute squared error over coordinates active for each target type.

    ``prediction`` and ``target`` have shape ``(..., N, 6)`` and ``types`` has
    shape ``(..., N)``. Inactive control points and coordinates belonging to
    ``CLOSE``, ``END``, or ``PAD`` do not contribute to the loss.
    """
    if prediction.shape != target.shape:
        msg = (
            "prediction and target must have the same shape, "
            f"got {tuple(prediction.shape)} and {tuple(target.shape)}"
        )
        raise ValueError(msg)
    if prediction.ndim == 0 or prediction.shape[-1] != COORD_DIM:
        msg = f"prediction and target must have shape (..., N, {COORD_DIM})"
        raise ValueError(msg)
    if types.shape != target.shape[:-1]:
        msg = (
            "types shape must match target without its last dimension, "
            f"got {tuple(types.shape)} and {tuple(target.shape)}"
        )
        raise ValueError(msg)

    mask = _active_coordinate_mask(types)
    difference = torch.where(mask, prediction - target, 0)
    loss = difference.square()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / mask.sum().clamp_min(1)
    msg = f"{reduction!r} is not a valid value for reduction"
    raise ValueError(msg)


def outline_loss(  # noqa: PLR0913
    type_logits: Tensor,
    coordinate_prediction: Tensor,
    target_types: Tensor,
    target_coords: Tensor,
    *,
    type_weight: float = 1.0,
    coordinate_weight: float = 1.0,
) -> Tensor:
    """Combine element-type classification and active-coordinate regression.

    Type loss is averaged over non-padding elements. Coordinate loss is
    averaged independently over coordinate scalars active for the target
    element types, then the two means are combined using the given weights.
    """
    if type_logits.shape[:-1] != target_types.shape:
        msg = (
            "type_logits shape without its last dimension must match "
            "target_types, "
            f"got {tuple(type_logits.shape)} and {tuple(target_types.shape)}"
        )
        raise ValueError(msg)
    if type_logits.ndim == 0 or type_logits.shape[-1] != TYPE_DIM:
        msg = f"type_logits must have shape (..., N, {TYPE_DIM})"
        raise ValueError(msg)

    type_loss = _functional.cross_entropy(
        type_logits.reshape(-1, TYPE_DIM),
        target_types.reshape(-1),
        ignore_index=ElementType.PAD.value,
        reduction="sum",
    )
    type_count = (target_types != ElementType.PAD.value).sum().clamp_min(1)
    type_loss = type_loss / type_count
    coords_loss = coordinate_loss(
        coordinate_prediction,
        target_coords,
        target_types,
    )
    return type_weight * type_loss + coordinate_weight * coords_loss


__all__ = ["coordinate_loss", "outline_loss"]
