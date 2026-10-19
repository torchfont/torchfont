from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from torchfont._outline import Outline


class NormalizeWinding(Transform):
    """Normalize independently reversible groups of contours.

    See :func:`~torchfont.transforms.functional.normalize_winding` for grouping
    and the ``clockwise`` convention.
    """

    def __init__(self, *, clockwise: bool = True) -> None:
        super().__init__()
        self.clockwise = clockwise

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.normalize_winding(inpt, clockwise=self.clockwise)


class RandomWinding(Transform):
    """Choose clockwise or counterclockwise for each group with equal probability.

    Grouping and direction follow :class:`NormalizeWinding`; relative winding
    within each group is preserved. Groups containing open subpaths or whose
    summed signed area is zero within floating-point roundoff are unchanged.
    Corresponding groups in one call share the sampled directions.
    """

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"clockwise": torch.randint(2, (length,), dtype=torch.bool)}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.normalize_winding(inpt, clockwise=params["clockwise"])


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
    """Randomly simplify groups with overlapping non-zero filled interiors.

    Groups contain closed contours connected by interior overlap, including
    containment. Point and shared-edge contacts alone do not join groups.

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


__all__ = [
    "NormalizeWinding",
    "RandomRemoveOverlaps",
    "RandomWinding",
    "RemoveOverlaps",
]
