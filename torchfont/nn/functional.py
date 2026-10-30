"""Loss functions for outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import torch
from torch.nn import functional as _functional

from torchfont._outline import (
    ElementType,
    _active_coordinate_mask,
    _validate_outline_tensors,
)

if TYPE_CHECKING:
    from torch import Tensor

_Reduction = Literal["none", "mean", "sum"]
_NUM_ELEMENT_TYPES = len(ElementType)


def _coordinate_loss(
    prediction: Tensor,
    target_types: Tensor,
    target_coords: Tensor,
    coordinate_loss: Literal["l1", "l2", "smooth_l1"],
    reduction: _Reduction,
) -> Tensor:
    if prediction.shape != target_coords.shape:
        msg = (
            "prediction must have the same shape as the target coordinates, "
            f"got {tuple(prediction.shape)} and {tuple(target_coords.shape)}"
        )
        raise ValueError(msg)
    _validate_outline_tensors(target_types, target_coords)

    mask = _active_coordinate_mask(target_types)
    difference = torch.where(mask, prediction - target_coords, 0)
    if coordinate_loss == "l1":
        loss = difference.abs()
    elif coordinate_loss == "l2":
        loss = difference.square()
    elif coordinate_loss == "smooth_l1":
        loss = _functional.smooth_l1_loss(
            difference, torch.zeros_like(difference), reduction="none"
        )
    else:
        msg = f"{coordinate_loss!r} is not a valid value for coordinate_loss"
        raise ValueError(msg)
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
    coordinate_loss: Literal["l1", "l2", "smooth_l1"] = "l2",
    type_weight: float = 1.0,
    coordinate_weight: float = 100.0,
    reduction: _Reduction = "mean",
) -> Tensor:
    """Combine element-type classification and active-coordinate regression.

    ``coordinate_loss`` selects squared error (``"l2"``, the default),
    absolute error (``"l1"``), or smooth L1 error (``"smooth_l1"``,
    with beta 1) for active coordinate scalars.

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
    coords_loss = _coordinate_loss(
        coordinate_prediction, target_types, target_coords, coordinate_loss, reduction
    )
    if reduction == "none":
        return type_weight * type_loss.sum(
            dim=-1
        ) + coordinate_weight * coords_loss.sum(dim=(-2, -1))
    if reduction == "sum":
        return type_weight * type_loss.sum() + coordinate_weight * coords_loss
    if reduction == "mean":
        type_count = (target_types != ElementType.PAD.value).sum().clamp_min(1)
        return (
            type_weight * type_loss.sum() / type_count + coordinate_weight * coords_loss
        )
    msg = f"{reduction!r} is not a valid value for reduction"
    raise ValueError(msg)


__all__ = ["outline_loss"]
