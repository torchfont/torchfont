"""Neural network building blocks for font outlines."""

from torchfont.nn import functional
from torchfont.nn._embedding import OutlineEmbedding
from torchfont.nn._loss import OutlineLoss

__all__ = ["OutlineEmbedding", "OutlineLoss", "functional"]
