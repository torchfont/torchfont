"""Neural network embeddings for outline tensors."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from torchfont._outline import _COORD_DIM, ElementType
from torchfont.nn._utils import _active_coordinate_mask, _validate_outline_tensors

_NUM_ELEMENT_TYPES = len(ElementType)


class OutlineEmbedding(nn.Module):
    """Embed element types and continuous coordinates into token features.

    ``types`` has shape ``(..., N)`` and ``coords`` has shape ``(..., N, 6)``,
    so the output has shape ``(..., N, embedding_dim)``. Padding tokens produce
    zero vectors.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the element-type embedding and coordinate projection."""
        super().__init__()
        self.embedding_dim = embedding_dim
        self.type_embedding = nn.Embedding(
            _NUM_ELEMENT_TYPES,
            embedding_dim,
            padding_idx=ElementType.PAD.value,
            device=device,
            dtype=dtype,
        )
        self.coord_projection = nn.Linear(
            _COORD_DIM,
            embedding_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw both branches from one uniform bound over the combined width."""
        bound = 1 / math.sqrt(_NUM_ELEMENT_TYPES + _COORD_DIM)
        nn.init.uniform_(self.type_embedding.weight, -bound, bound)
        nn.init.uniform_(self.coord_projection.weight, -bound, bound)
        with torch.no_grad():
            self.type_embedding.weight[ElementType.PAD.value].zero_()

    def forward(self, types: Tensor, coords: Tensor) -> Tensor:
        """Return the sum of element-type and coordinate embeddings."""
        _validate_outline_tensors(types, coords)
        active_coords = torch.where(_active_coordinate_mask(types), coords, 0)
        embedded = self.type_embedding(types) + self.coord_projection(active_coords)
        return torch.where((types != ElementType.PAD.value).unsqueeze(-1), embedded, 0)

    def extra_repr(self) -> str:
        """Return the module configuration for :func:`repr`."""
        return f"embedding_dim={self.embedding_dim}"


__all__ = ["OutlineEmbedding"]
