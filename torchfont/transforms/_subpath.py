"""Subpath transforms."""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING, Any, ClassVar, SupportsIndex

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


class TruncateSubpaths(Transform):
    """Keep the longest whole-subpath prefix within the given limits."""

    def __init__(
        self,
        max_length: SupportsIndex | None = None,
        max_subpaths: SupportsIndex | None = None,
    ) -> None:
        super().__init__()
        self.max_length = None if max_length is None else index(max_length)
        self.max_subpaths = None if max_subpaths is None else index(max_subpaths)
        if self.max_length is None and self.max_subpaths is None:
            msg = "max_length or max_subpaths must be specified"
            raise ValueError(msg)
        if self.max_length is not None and self.max_length < 1:
            msg = "max_length must be positive"
            raise ValueError(msg)
        if self.max_subpaths is not None and self.max_subpaths < 0:
            msg = "max_subpaths must be non-negative"
            raise ValueError(msg)

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.truncate_subpaths(
            inpt,
            self.max_length,
            self.max_subpaths,
        )


class RandomTruncateSubpaths(Transform):
    """Randomly drop subpaths until the given limits are met."""

    def __init__(
        self,
        max_length: SupportsIndex | None = None,
        max_subpaths: SupportsIndex | None = None,
    ) -> None:
        super().__init__()
        self.max_length = None if max_length is None else index(max_length)
        self.max_subpaths = None if max_subpaths is None else index(max_subpaths)
        if self.max_length is None and self.max_subpaths is None:
            msg = "max_length or max_subpaths must be specified"
            raise ValueError(msg)
        if self.max_length is not None and self.max_length < 1:
            msg = "max_length must be positive"
            raise ValueError(msg)
        if self.max_subpaths is not None and self.max_subpaths < 0:
            msg = "max_subpaths must be non-negative"
            raise ValueError(msg)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"removal_values": torch.rand(length)}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.drop_subpaths_to_fit(
            inpt,
            params["removal_values"],
            self.max_length,
            self.max_subpaths,
        )


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
    "RandomTruncateSubpaths",
    "SplitSubpaths",
    "TruncateSubpaths",
]
