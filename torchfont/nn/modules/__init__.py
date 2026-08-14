"""Neural network modules for font outlines."""

from torchfont.nn.modules.embedding import OutlineEmbedding
from torchfont.nn.modules.loss import OutlineLoss

__all__ = ["OutlineEmbedding", "OutlineLoss"]
