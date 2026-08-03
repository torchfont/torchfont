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


class _RandomSubpathTransform(Transform):
    function: ClassVar[Callable[..., Outline]]

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"values": torch.rand(length)}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return self.function(inpt, params["values"])


class RandomizeSubpathStartPoints(_RandomSubpathTransform):
    """Choose a random start point for every closed subpath."""

    function = staticmethod(_functional.set_subpath_start_points)


class RandomizeSubpathOrder(_RandomSubpathTransform):
    """Randomly permute whole subpaths."""

    function = staticmethod(_functional.reorder_subpaths)


__all__ = [
    "NormalizeSubpathStartPoints",
    "RandomizeSubpathOrder",
    "RandomizeSubpathStartPoints",
]
