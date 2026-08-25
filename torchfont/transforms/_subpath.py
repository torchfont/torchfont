"""Subpath transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, ClassVar

import torch

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchfont._outline import Outline


class NormalizeSubpathStartPoints(Transform):
    """Choose a deterministic start point for each closed subpath."""

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.normalize_subpath_start_points(inpt)


class SplitSubpaths(Transform):
    """Split each outline into a tuple of independently encoded subpaths."""

    def transform(self, inpt: Outline, params: dict[str, Any]) -> tuple[Outline, ...]:
        del params
        return _functional.split_subpaths(inpt)


class RandomSubpathDropout(Transform):
    """Drop each subpath independently with probability ``p``."""

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if not 0.0 <= p <= 1.0:
            msg = "p must be between 0 and 1"
            raise ValueError(msg)
        self.p = p

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"drop_mask": torch.rand(length) < self.p}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.drop_subpaths(
            inpt,
            params["drop_mask"],
        )


class _RandomSubpathTransform(Transform):
    function: ClassVar[Callable[..., Outline]]

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"values": torch.rand(length)}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return self.function(inpt, params["values"])


class RandomSubpathStartPoints(_RandomSubpathTransform):
    """Choose a random start point for every closed subpath."""

    function = staticmethod(_functional.set_subpath_start_points)


class RandomSubpathOrder(_RandomSubpathTransform):
    """Randomly permute whole subpaths."""

    function = staticmethod(_functional.reorder_subpaths)


__all__ = [
    "NormalizeSubpathStartPoints",
    "RandomSubpathDropout",
    "RandomSubpathOrder",
    "RandomSubpathStartPoints",
    "SplitSubpaths",
]
