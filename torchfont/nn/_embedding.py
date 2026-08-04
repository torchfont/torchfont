"""Neural network embeddings for outline tensors."""

from __future__ import annotations

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
