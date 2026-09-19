"""Whole-outline transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from torchfont._outline import Outline


class RemoveOverlaps(Transform):
    """Merge overlapping subpaths.

    Args:
        verify: Whether to preserve the input when coverage changes.
        verify_size: Verification resolution in pixels. Must be between 1 and
            4096.

    """

    def __init__(self, *, verify: bool = False, verify_size: int = 256) -> None:
        super().__init__()
        self.verify = verify
        self.verify_size = verify_size

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.remove_overlaps(
            inpt, verify=self.verify, verify_size=self.verify_size
        )


class RandomRemoveOverlaps(Transform):
    """Randomly simplify bbox-connected overlap groups.

    Args:
        verify: Whether to preserve the input when coverage changes.
        verify_size: Verification resolution in pixels. Must be between 1 and
            4096.

    """

    def __init__(self, *, verify: bool = False, verify_size: int = 256) -> None:
        super().__init__()
        self.verify = verify
        self.verify_size = verify_size

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"values": torch.rand(length)}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.remove_overlap_groups(
            inpt,
            params["values"],
            verify=self.verify,
            verify_size=self.verify_size,
        )


__all__ = ["RandomRemoveOverlaps", "RemoveOverlaps"]
