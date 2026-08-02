"""Semantic type conversion transforms."""

from typing import Any

from torch import Tensor

from torchfont.tf_tensors import TFTensor
from torchfont.transforms._transform import Transform


class ToPureTensor(Transform):
    """Remove TorchFont semantic tensor subclasses without copying data."""

    _transformed_types = (TFTensor,)

    def transform(self, inpt: TFTensor, params: dict[str, Any]) -> Tensor:
        del params
        return inpt.as_subclass(Tensor)


__all__ = ["ToPureTensor"]
