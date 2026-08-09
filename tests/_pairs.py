"""Tensor-pair adapters for kernels whose public API takes an ``Outline``.

Many kernel tests were written against ``(types, coords)`` pairs. The public
functional API takes and returns :class:`torchfont.Outline`, so these adapters
keep those tests focused on kernel behaviour instead of container plumbing.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont import Outline
from torchfont.transforms import functional as F  # noqa: N812

if TYPE_CHECKING:
    from torch import Tensor


def _pair(outline: Outline) -> tuple[Tensor, Tensor]:
    return outline.types, outline.coords


def quad_to_cubic(
    types: Tensor,
    coords: Tensor,
    merge_curves: bool = False,  # noqa: FBT001, FBT002
) -> tuple[Tensor, Tensor]:
    """Call :func:`torchfont.transforms.functional.quad_to_cubic` on a pair."""
    return _pair(F.quad_to_cubic(Outline(types, coords), merge_curves=merge_curves))


def cubic_to_quad(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Call :func:`torchfont.transforms.functional.cubic_to_quad` on a pair."""
    return _pair(F.cubic_to_quad(Outline(types, coords)))


def merge_curves(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Call :func:`torchfont.transforms.functional.merge_curves` on a pair."""
    return _pair(F.merge_curves(Outline(types, coords)))


def remove_overlaps(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Call :func:`torchfont.transforms.functional.remove_overlaps` on a pair."""
    return _pair(F.remove_overlaps(Outline(types, coords)))


def normalize_subpath_start_points(
    types: Tensor, coords: Tensor
) -> tuple[Tensor, Tensor]:
    """Call the subpath start-point normaliser on a pair."""
    return _pair(F.normalize_subpath_start_points(Outline(types, coords)))
