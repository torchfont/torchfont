"""Transform composition containers."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast

import torch
from torch import nn

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


class RandomChoice(nn.Module):
    """Apply one transform randomly selected from a sequence."""

    def __init__(
        self,
        transforms: Iterable[nn.Module] | nn.ModuleList,
        p: list[float] | None = None,
    ) -> None:
        super().__init__()
        self.transforms = _module_list(transforms)
        if not self.transforms:
            msg = "transforms must contain at least one transform"
            raise ValueError(msg)
        if p is None:
            p = [1.0] * len(self.transforms)
        elif len(p) != len(self.transforms):
            msg = (
                "p length must match the number of transforms, "
                f"got {len(p)} and {len(self.transforms)}"
            )
            raise ValueError(msg)
        if not all(
            math.isfinite(probability) and probability >= 0.0 for probability in p
        ):
            msg = "p values must be non-negative and finite"
            raise ValueError(msg)
        total = sum(p)
        if total == 0.0:
            msg = "p values must have a positive sum"
            raise ValueError(msg)
        self.p = [probability / total for probability in p]

    def forward(self, *inputs: object) -> object:
        """Apply the randomly selected transform to all inputs."""
        index = int(torch.multinomial(torch.tensor(self.p), 1).item())
        return self.transforms[index](*inputs)

    def extra_repr(self) -> str:
        return f"p={self.p}"


class RandomOrder(nn.Module):
    """Apply a sequence of transforms in random order."""

    def __init__(
        self,
        transforms: Iterable[nn.Module] | nn.ModuleList,
    ) -> None:
        super().__init__()
        self.transforms = _module_list(transforms)
        if not self.transforms:
            msg = "transforms must contain at least one transform"
            raise ValueError(msg)

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms in a randomly sampled order."""
        unpack = len(inputs) > 1
        output: object = inputs if unpack else inputs[0]
        for transform_index in torch.randperm(len(self.transforms)):
            output = self.transforms[transform_index](*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output


__all__ = ["Compose", "RandomApply", "RandomChoice", "RandomOrder"]
