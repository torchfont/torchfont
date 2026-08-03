"""Transform composition containers."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import nn

from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from collections.abc import Iterable


def _module_list(
    transforms: Iterable[nn.Module] | nn.ModuleList,
) -> nn.ModuleList:
    return (
        transforms
        if isinstance(transforms, nn.ModuleList)
        else nn.ModuleList(transforms)
    )


class Compose(nn.Module):
    """Apply a sequence of transforms in order."""

    def __init__(
        self,
        transforms: Iterable[nn.Module] | nn.ModuleList,
    ) -> None:
        super().__init__()
        self.transforms = _module_list(transforms)

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms to the inputs."""
        unpack = len(inputs) > 1
        output: object = inputs if unpack else inputs[0]
        for transform in self.transforms:
            output = transform(*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output


class RandomApply(nn.Module):
    """Apply one transform with probability ``p``."""

    def __init__(
        self,
        transform: nn.Module,
        p: float = 0.5,
    ) -> None:
        super().__init__()
        if not isinstance(transform, nn.Module):
            msg = "transform must be an nn.Module"
            raise TypeError(msg)
        self.transform = transform
        if not 0.0 <= p <= 1.0:
            msg = "p must be between 0 and 1"
            raise ValueError(msg)
        self.p = p

    def forward(self, *inputs: object) -> object:
        """Apply the configured transform or return inputs unchanged."""
        unpack = len(inputs) > 1
        if torch.rand(()) >= self.p:
            return inputs if unpack else inputs[0]
        return self.transform(*inputs)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class SameParams(nn.Module):
    """Apply one transform with parameters shared across semantic leaves."""

    def __init__(self, transform: Transform) -> None:
        super().__init__()
        if not isinstance(transform, Transform):
            msg = "transform must be a Transform"
            raise TypeError(msg)
        self.transform = transform

    def forward(self, *inputs: object) -> object:
        """Apply the wrapped transform with one shared parameter sample."""
        return self.transform(*inputs, _same_params=True)


__all__ = ["Compose", "RandomApply", "SameParams"]
