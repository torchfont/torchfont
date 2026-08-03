"""Transform composition containers."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import cast

import torch
from torch import nn

from torchfont.transforms._transform import Transform


class Compose(Transform):
    """Apply a sequence of transforms in order."""

    def __init__(
        self,
        transforms: Sequence[Callable[..., object]] | nn.ModuleList,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, (Sequence, nn.ModuleList)):
            msg = "transforms must be a sequence of callables or an nn.ModuleList"
            raise TypeError(msg)
        if not transforms:
            msg = "transforms must not be empty"
            raise ValueError(msg)
        self.transforms = transforms

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms to the inputs."""
        unpack = len(inputs) > 1
        for transform in self.transforms:
            output = transform(*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output

    def extra_repr(self) -> str:
        return "\n".join(f"    {transform}" for transform in self.transforms)


class RandomApply(Transform):
    """Apply a sequence of transforms with probability ``p``."""

    def __init__(
        self,
        transforms: Sequence[Callable[..., object]] | nn.ModuleList,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        if not isinstance(transforms, (Sequence, nn.ModuleList)):
            msg = "transforms must be a sequence of callables or an nn.ModuleList"
            raise TypeError(msg)
        if not transforms:
            msg = "transforms must not be empty"
            raise ValueError(msg)
        self.transforms = transforms
        if not 0.0 <= p <= 1.0:
            msg = "p must be between 0 and 1"
            raise ValueError(msg)
        self.p = p

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms together or return inputs unchanged."""
        unpack = len(inputs) > 1
        if torch.rand(()) >= self.p:
            return inputs if unpack else inputs[0]
        for transform in self.transforms:
            output = transform(*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output

    def extra_repr(self) -> str:
        return "\n".join(f"    {transform}" for transform in self.transforms)


__all__ = ["Compose", "RandomApply"]
