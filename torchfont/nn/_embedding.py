"""Neural network embeddings for outline tensors."""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn

from torchfont._outline import COORD_DIM, TYPE_DIM, ElementType
from torchfont.nn._utils import _active_coordinate_mask


class OutlineEmbedding(nn.Module):
    """Embed element types and continuous coordinates into token features.

    The input tensors may be unbatched or have any number of leading batch
    dimensions. Their shapes are ``(..., N)`` and ``(..., N, 6)``; the output
    has shape ``(..., N, embedding_dim)``. Padding tokens produce zero vectors.
    """

    def __init__(
        self,
        embedding_dim: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        super().__init__()
        self.embedding_dim = embedding_dim
        self.type_embedding = nn.Embedding(
            TYPE_DIM,
            embedding_dim,
            padding_idx=ElementType.PAD.value,
            device=device,
            dtype=dtype,
        )
        self.coord_projection = nn.Linear(
            COORD_DIM,
            embedding_dim,
            bias=False,
            device=device,
            dtype=dtype,
        )
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Draw both branches from one uniform bound over the combined width."""
        bound = 1 / math.sqrt(TYPE_DIM + COORD_DIM)
        nn.init.uniform_(self.type_embedding.weight, -bound, bound)
        nn.init.uniform_(self.coord_projection.weight, -bound, bound)
        with torch.no_grad():
            self.type_embedding.weight[ElementType.PAD.value].zero_()

    def forward(self, types: Tensor, coords: Tensor) -> Tensor:
        """Return the sum of element-type and coordinate embeddings."""
        if types.shape != coords.shape[:-1]:
            msg = (
                "types shape must match coords without its last dimension, "
                f"got {tuple(types.shape)} and {tuple(coords.shape)}"
            )
            raise ValueError(msg)
        active_coords = torch.where(_active_coordinate_mask(types), coords, 0)
        embedded = self.type_embedding(types) + self.coord_projection(active_coords)
        return torch.where((types != ElementType.PAD.value).unsqueeze(-1), embedded, 0)

    def extra_repr(self) -> str:
        return f"embedding_dim={self.embedding_dim}"


__all__ = ["OutlineEmbedding"]
