"""Geometric transforms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from torchfont._outline import Outline


class HorizontalFlip(Transform):
    """Flip outlines horizontally."""

    def __init__(self, *, preserve_winding: bool = True) -> None:
        super().__init__()
        self.preserve_winding = preserve_winding

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.horizontal_flip(inpt, preserve_winding=self.preserve_winding)


class VerticalFlip(Transform):
    """Flip outlines vertically."""

    def __init__(self, *, preserve_winding: bool = True) -> None:
        super().__init__()
        self.preserve_winding = preserve_winding

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.vertical_flip(inpt, preserve_winding=self.preserve_winding)


class RandomHorizontalFlip(HorizontalFlip):
    """Flip each outline independently with probability ``p``.

    Wrap with :class:`SameParams` to share one decision across outlines.
    """

    def __init__(self, p: float = 0.5, *, preserve_winding: bool = True) -> None:
        super().__init__(preserve_winding=preserve_winding)
        if not 0.0 <= p <= 1.0:
            msg = "p must be between 0 and 1"
            raise ValueError(msg)
        self.p = p

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        return {"apply": torch.rand(()).item() < self.p}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return super().transform(inpt, params) if params["apply"] else inpt


class RandomVerticalFlip(VerticalFlip):
    """Flip each outline independently with probability ``p``.

    Wrap with :class:`SameParams` to share one decision across outlines.
    """

    def __init__(self, p: float = 0.5, *, preserve_winding: bool = True) -> None:
        super().__init__(preserve_winding=preserve_winding)
        if not 0.0 <= p <= 1.0:
            msg = "p must be between 0 and 1"
            raise ValueError(msg)
        self.p = p

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        return {"apply": torch.rand(()).item() < self.p}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return super().transform(inpt, params) if params["apply"] else inpt


class Affine(Transform):
    """Apply a fixed affine transformation."""

    def __init__(
        self,
        *,
        angle: float = 0.0,
        translate: tuple[float, float] = (0.0, 0.0),
        scale: float = 1.0,
        shear: float = 0.0,
    ) -> None:
        super().__init__()
        self.angle = angle
        self.translate = translate
        self.scale = scale
        self.shear = shear

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.affine(
            inpt,
            angle=self.angle,
            translate=self.translate,
            scale=self.scale,
            shear=self.shear,
        )


def _symmetric_range(value: float | tuple[float, float]) -> tuple[float, float]:
    if isinstance(value, (float, int)):
        bounds = (-abs(float(value)), abs(float(value)))
    else:
        lower, upper = value
        bounds = (float(lower), float(upper))
    if not all(math.isfinite(item) for item in bounds) or bounds[0] > bounds[1]:
        msg = "range values must be finite and ordered"
        raise ValueError(msg)
    return bounds


class RandomAffine(Transform):
    """Apply independently sampled affine parameters to each outline.

    Wrap with :class:`SameParams` to share one parameter sample across outlines.
    """

    def __init__(
        self,
        *,
        degrees: float | tuple[float, float] = 0.0,
        translate: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        shear: float | tuple[float, float] = 0.0,
    ) -> None:
        super().__init__()
        self.degrees = _symmetric_range(degrees)
        if translate is not None:
            translate_x, translate_y = translate
            translate = (float(translate_x), float(translate_y))
            if not all(math.isfinite(value) and value >= 0.0 for value in translate):
                msg = "translate values must be non-negative and finite"
                raise ValueError(msg)
        if scale is not None:
            scale_min, scale_max = scale
            scale = (float(scale_min), float(scale_max))
            if not all(math.isfinite(value) and value > 0.0 for value in scale):
                msg = "scale values must be positive and finite"
                raise ValueError(msg)
            if scale[0] > scale[1]:
                msg = "scale values must be ordered"
                raise ValueError(msg)
        self.translate = translate
        self.scale = scale
        self.shear = _symmetric_range(shear)

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        values = torch.rand(5)

        def uniform(bounds: tuple[float, float], index: int) -> float:
            return bounds[0] + (bounds[1] - bounds[0]) * values[index].item()

        translate = self.translate or (0.0, 0.0)
        return {
            "angle": uniform(self.degrees, 0),
            "translate": (
                uniform((-translate[0], translate[0]), 1),
                uniform((-translate[1], translate[1]), 2),
            ),
            "scale": uniform(self.scale, 3) if self.scale is not None else 1.0,
            "shear": uniform(self.shear, 4),
        }

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.affine(inpt, **params)


class RandomCoordJitter(Transform):
    """Add Gaussian coordinate noise to active outline elements."""

    def __init__(self, std: float) -> None:
        super().__init__()
        if not math.isfinite(std) or std < 0.0:
            msg = "std must be non-negative and finite"
            raise ValueError(msg)
        self.std = std

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.coords.size(0) for inpt in flat_inputs), default=0)
        return {"noise": torch.randn((length, 3, 2)) * self.std}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.coord_jitter(inpt, params["noise"])


__all__ = [
    "Affine",
    "HorizontalFlip",
    "RandomAffine",
    "RandomCoordJitter",
    "RandomHorizontalFlip",
    "RandomVerticalFlip",
    "VerticalFlip",
]
