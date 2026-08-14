"""Neural network building blocks for font outlines."""

from torchfont.nn import functional
from torchfont.nn.modules import OutlineEmbedding, OutlineLoss

__all__ = ["OutlineEmbedding", "OutlineLoss", "functional"]
