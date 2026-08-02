"""Curve and segment transforms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, ClassVar, cast

import torch

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchfont.structures import Outline


class _SimpleCurveTransform(Transform):
    function: ClassVar[Callable[..., object]]

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return cast("Outline", self._call_kernel(self.function, inpt))


class QuadToCubic(Transform):
    """Convert quadratic segments to cubic segments."""

    def __init__(self, *, merge_curves: bool = False) -> None:
        super().__init__()
        self.merge_curves = merge_curves

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return cast(
            "Outline",
            self._call_kernel(
                _functional.quad_to_cubic, inpt, merge_curves=self.merge_curves
            ),
        )


class CubicToQuad(_SimpleCurveTransform):
    """Convert cubic segments to quadratic segments."""

    function = staticmethod(_functional.cubic_to_quad)


class MergeCurves(_SimpleCurveTransform):
    """Merge adjacent pieces of the same parent segment."""

    function = staticmethod(_functional.merge_curves)


class RandomSplitSegments(Transform):
    """Randomly split line and Bezier segments without changing their shape."""

    def __init__(
        self,
        split_probability: float = 0.2,
        split_range: tuple[float, float] = (0.2, 0.8),
    ) -> None:
        super().__init__()
        if not 0.0 <= split_probability <= 1.0:
            msg = "split_probability must be between 0 and 1"
            raise ValueError(msg)
        if not (
            0.0 < split_range[0] <= split_range[1] < 1.0
            and all(math.isfinite(value) for value in split_range)
        ):
            msg = "split_range must satisfy 0 < min <= max < 1"
            raise ValueError(msg)
        self.split_probability = split_probability
        self.split_range = split_range

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"values": torch.rand((2, length))}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        values = params["values"]
        return cast(
            "Outline",
            self._call_kernel(
                _functional.split_segments,
                inpt,
                values[0],
                values[1],
                split_probability=self.split_probability,
                split_range=self.split_range,
            ),
        )


__all__ = ["CubicToQuad", "MergeCurves", "QuadToCubic", "RandomSplitSegments"]
