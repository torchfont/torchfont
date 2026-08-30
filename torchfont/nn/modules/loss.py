"""Loss modules for outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch import nn

from torchfont.nn import functional

if TYPE_CHECKING:
    from torch import Tensor


class OutlineLoss(nn.Module):
    """Combine element-type and active-coordinate losses.

    ``reduction`` controls aggregation across outlines. Padding elements do not
    contribute.
    """

    def __init__(
        self,
        *,
        type_weight: float = 1.0,
        coordinate_weight: float = 100.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        """Initialize the weights and reduction applied to the loss."""
        super().__init__()
        self.type_weight = type_weight
        self.coordinate_weight = coordinate_weight
        self.reduction = reduction

    def forward(
        self,
        type_logits: Tensor,
        coordinate_prediction: Tensor,
        target_types: Tensor,
        target_coords: Tensor,
    ) -> Tensor:
        """Return the weighted mean outline loss."""
        return functional.outline_loss(
            type_logits,
            coordinate_prediction,
            target_types,
            target_coords,
            type_weight=self.type_weight,
            coordinate_weight=self.coordinate_weight,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        """Return the module configuration for :func:`repr`."""
        return (
            f"type_weight={self.type_weight}, "
            f"coordinate_weight={self.coordinate_weight}, "
            f"reduction={self.reduction!r}"
        )


__all__ = ["OutlineLoss"]
