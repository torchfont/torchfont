"""Loss functions for outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional as _functional

from torchfont._outline import ElementType
from torchfont.nn._utils import _active_coordinate_mask, _validate_outline_tensors

if TYPE_CHECKING:
    from torch import Tensor

_Reduction = Literal["none", "mean", "sum"]
_NUM_ELEMENT_TYPES = len(ElementType)


def coordinate_mse_loss(
    prediction: Tensor,
    target_types: Tensor,
    target_coords: Tensor,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Compute squared error over coordinates active for each target type.

    ``prediction`` and ``target_coords`` have shape ``(..., N, 6)`` while
    ``target_types`` has shape ``(..., N)``. Inactive control points and
    coordinates belonging to ``CLOSE``, ``END``, or ``PAD`` do not contribute.
    """
    if prediction.shape != target_coords.shape:
        msg = (
            "prediction must have the same shape as the target coordinates, "
            f"got {tuple(prediction.shape)} and {tuple(target_coords.shape)}"
        )
        raise ValueError(msg)
    _validate_outline_tensors(target_types, target_coords)

    mask = _active_coordinate_mask(target_types)
    difference = torch.where(mask, prediction - target_coords, 0)
    loss = difference.square()
    if reduction == "none":
        return loss
    if reduction == "sum":
        return loss.sum()
    if reduction == "mean":
        return loss.sum() / mask.sum().clamp_min(1)
    msg = f"{reduction!r} is not a valid value for reduction"
    raise ValueError(msg)


def outline_loss(
    type_logits: Tensor,
    coordinate_prediction: Tensor,
    target_types: Tensor,
    target_coords: Tensor,
    *,
    type_weight: float = 1.0,
    coordinate_weight: float = 100.0,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Combine element-type classification and active-coordinate regression.

    With ``reduction="none"``, losses are summed within each outline and the
    result has shape ``target_types.shape[:-1]``. ``"sum"`` sums those outline
    losses. ``"mean"`` averages type loss over all non-padding elements and
    coordinate loss independently over all active coordinate scalars before
    combining them using the given weights.
    """
    if type_logits.shape[:-1] != target_types.shape:
        msg = (
            "type_logits shape without its last dimension must match the target "
            "element types, "
            f"got {tuple(type_logits.shape)} and {tuple(target_types.shape)}"
        )
        raise ValueError(msg)
    if type_logits.shape[-1] != _NUM_ELEMENT_TYPES:
        msg = f"type_logits must have shape (..., N, {_NUM_ELEMENT_TYPES})"
        raise ValueError(msg)

    type_loss = _functional.cross_entropy(
        type_logits.reshape(-1, _NUM_ELEMENT_TYPES),
        target_types.reshape(-1),
        ignore_index=ElementType.PAD.value,
        reduction="none",
    ).reshape(target_types.shape)
    coords_loss = coordinate_mse_loss(
        coordinate_prediction, target_types, target_coords, reduction="none"
    )
    if reduction == "none":
        return type_weight * type_loss.sum(
            dim=-1
        ) + coordinate_weight * coords_loss.sum(dim=(-2, -1))
    if reduction == "sum":
        return type_weight * type_loss.sum() + coordinate_weight * coords_loss.sum()
    if reduction == "mean":
        type_count = (target_types != ElementType.PAD.value).sum().clamp_min(1)
        coordinate_count = _active_coordinate_mask(target_types).sum().clamp_min(1)
        return (
            type_weight * type_loss.sum() / type_count
            + coordinate_weight * coords_loss.sum() / coordinate_count
        )
    msg = f"{reduction!r} is not a valid value for reduction"
    raise ValueError(msg)


__all__ = ["coordinate_mse_loss", "outline_loss"]
