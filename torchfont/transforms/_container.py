"""Transform composition containers."""

from __future__ import annotations

from collections.abc import Sequence
from typing import cast

import torch
from torch import nn

from torchfont.transforms._transform import Transform


def _module_list(
    transforms: Sequence[nn.Module] | nn.ModuleList,
) -> nn.ModuleList:
    if not isinstance(transforms, (Sequence, nn.ModuleList)):
        msg = "transforms must be a sequence of nn.Module objects"
        raise TypeError(msg)
    if not transforms:
        msg = "transforms must not be empty"
        raise ValueError(msg)
    invalid = next(
        (transform for transform in transforms if not isinstance(transform, nn.Module)),
        None,
    )
    if invalid is not None:
        msg = "transforms must contain only nn.Module objects"
        raise TypeError(msg)
    return (
        transforms
        if isinstance(transforms, nn.ModuleList)
        else nn.ModuleList(transforms)
    )


class Compose(nn.Module):
    """Apply a sequence of transforms in order."""

    def __init__(
        self,
        transforms: Sequence[nn.Module] | nn.ModuleList,
    ) -> None:
        super().__init__()
        self.transforms = _module_list(transforms)

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms to the inputs."""
        unpack = len(inputs) > 1
        for transform in self.transforms:
            output = transform(*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output

    def extra_repr(self) -> str:
        return ""


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
        return self.transform.forward_same_params(*inputs)


__all__ = ["Compose", "RandomApply", "SameParams"]
