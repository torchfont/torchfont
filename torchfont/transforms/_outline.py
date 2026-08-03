"""Whole-outline transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from torchfont._outline import Outline


class RemoveOverlaps(Transform):
    """Merge overlapping subpaths."""

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.remove_overlaps(inpt)


class RandomRemoveOverlaps(Transform):
    """Randomly simplify bbox-connected overlap groups."""

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"values": torch.rand(length)}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.remove_overlap_groups(inpt, params["values"])


__all__ = ["RandomRemoveOverlaps", "RemoveOverlaps"]
