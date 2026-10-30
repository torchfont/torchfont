"""Loss modules for outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

from torch import nn

from torchfont.nn import functional

if TYPE_CHECKING:
    from torch import Tensor


class OutlineLoss(nn.Module):
    """Combine element-type and active-coordinate losses.

    ``coordinate_loss`` selects squared error (``"l2"``, the default),
    absolute error (``"l1"``), or smooth L1 error (``"smooth_l1"``,
    with beta 1) for active coordinate scalars.
    ``reduction`` controls aggregation across outlines. Padding elements do not
    contribute.
    """

    def __init__(
        self,
        *,
        coordinate_loss: Literal["l1", "l2", "smooth_l1"] = "l2",
        type_weight: float = 1.0,
        coordinate_weight: float = 100.0,
        reduction: Literal["none", "mean", "sum"] = "mean",
    ) -> None:
        """Initialize the weights and reduction applied to the loss."""
        super().__init__()
        self.coordinate_loss = coordinate_loss
        self.type_weight = type_weight
        self.coordinate_weight = coordinate_weight
        self.reduction = reduction

    def forward(  # ty: ignore[invalid-mutable-override]
        self,
        type_logits: Tensor,
        coordinate_prediction: Tensor,
        target_types: Tensor,
        target_coords: Tensor,
    ) -> Tensor:
        """Compute outline loss using the configured weights and reduction."""
        return functional.outline_loss(
            type_logits,
            coordinate_prediction,
            target_types,
            target_coords,
            coordinate_loss=self.coordinate_loss,
            type_weight=self.type_weight,
            coordinate_weight=self.coordinate_weight,
            reduction=self.reduction,
        )

    def extra_repr(self) -> str:
        """Return the module configuration for :func:`repr`."""
        return (
            f"coordinate_loss={self.coordinate_loss!r}, "
            f"type_weight={self.type_weight}, "
            f"coordinate_weight={self.coordinate_weight}, "
            f"reduction={self.reduction!r}"
        )


__all__ = ["OutlineLoss"]
