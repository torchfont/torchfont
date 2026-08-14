"""Loss modules for outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torch import nn

from torchfont.nn import functional

if TYPE_CHECKING:
    from torch import Tensor

    from torchfont._outline import Outline


class OutlineLoss(nn.Module):
    """Combine element-type and active-coordinate losses.

    The two losses are averaged independently before ``type_weight`` and
    ``coordinate_weight`` are applied. Padding elements do not contribute.
    """

    def __init__(
        self,
        *,
        type_weight: float = 1.0,
        coordinate_weight: float = 1.0,
    ) -> None:
        """Initialize the weights applied to both loss components."""
        super().__init__()
        self.type_weight = type_weight
        self.coordinate_weight = coordinate_weight

    def forward(
        self,
        type_logits: Tensor,
        coordinate_prediction: Tensor,
        target: Outline,
    ) -> Tensor:
        """Return the weighted mean outline loss."""
        return functional.outline_loss(
            type_logits,
            coordinate_prediction,
            target,
            type_weight=self.type_weight,
            coordinate_weight=self.coordinate_weight,
        )

    def extra_repr(self) -> str:
        """Return the module configuration for :func:`repr`."""
        return (
            f"type_weight={self.type_weight}, "
            f"coordinate_weight={self.coordinate_weight}"
        )


__all__ = ["OutlineLoss"]
