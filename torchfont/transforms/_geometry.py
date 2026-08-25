"""Geometric transforms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch
from torch.nn import functional as nn_functional

from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

_SHEAR_RANGE_SIZE = 2
_XY_SHEAR_RANGE_SIZE = 4

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
    """Flip outlines with probability ``p`` using one shared decision."""

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
    """Flip outlines with probability ``p`` using one shared decision."""

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
        shear: float | tuple[float, float] = 0.0,
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
    """Apply one affine parameter sample to all outlines in the input."""

    def __init__(
        self,
        *,
        degrees: float | tuple[float, float] = 0.0,
        translate: tuple[float, float] | None = None,
        scale: tuple[float, float] | None = None,
        shear: float | tuple[float, float] | tuple[float, float, float, float] = 0.0,
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
        if isinstance(shear, (float, int)) or len(shear) == _SHEAR_RANGE_SIZE:
            self.shear = _symmetric_range(shear)
        elif len(shear) == _XY_SHEAR_RANGE_SIZE:
            self.shear = (*_symmetric_range(shear[:2]), *_symmetric_range(shear[2:]))
        else:
            msg = "shear must be a number or a sequence of two or four values"
            raise ValueError(msg)

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        values = torch.rand(6)

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
            "shear": (
                uniform(self.shear[:2], 4),
                uniform(self.shear[2:], 5)
                if len(self.shear) == _XY_SHEAR_RANGE_SIZE
                else 0.0,
            ),
        }

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.affine(inpt, **params)


class RandomRotation(Transform):
    """Rotate outlines by a randomly sampled angle."""

    def __init__(self, degrees: float | tuple[float, float]) -> None:
        super().__init__()
        self.degrees = _symmetric_range(degrees)

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        value = torch.rand(()).item()
        lower, upper = self.degrees
        return {"angle": lower + (upper - lower) * value}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.rotate(inpt, params["angle"])


def _positive_range(value: tuple[float, float], name: str) -> tuple[float, float]:
    lower, upper = value
    bounds = (float(lower), float(upper))
    if not all(math.isfinite(item) and item > 0.0 for item in bounds):
        msg = f"{name} values must be positive and finite"
        raise ValueError(msg)
    if bounds[0] > bounds[1]:
        msg = f"{name} values must be ordered"
        raise ValueError(msg)
    return bounds


class RandomScale(Transform):
    """Scale outlines independently along the x and y axes."""

    def __init__(
        self,
        scale_x: tuple[float, float] = (1.0, 1.0),
        scale_y: tuple[float, float] = (1.0, 1.0),
    ) -> None:
        super().__init__()
        self.scale_x = _positive_range(scale_x, "scale_x")
        self.scale_y = _positive_range(scale_y, "scale_y")

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        values = torch.rand(2)

        def uniform(bounds: tuple[float, float], index: int) -> float:
            return bounds[0] + (bounds[1] - bounds[0]) * values[index].item()

        return {
            "factors": (uniform(self.scale_x, 0), uniform(self.scale_y, 1)),
        }

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.scale(inpt, params["factors"])


class GaussianNoise(Transform):
    """Add Gaussian coordinate noise to active outline elements."""

    def __init__(self, sigma: float) -> None:
        super().__init__()
        if not math.isfinite(sigma) or sigma < 0.0:
            msg = "sigma must be non-negative and finite"
            raise ValueError(msg)
        self.sigma = sigma

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.coords.size(0) for inpt in flat_inputs), default=0)
        return {"noise": torch.randn((length, 3, 2)) * self.sigma}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.add_coordinate_noise(inpt, params["noise"])


def _pair(value: float | tuple[float, float], name: str) -> tuple[float, float]:
    if isinstance(value, (float, int)):
        pair = (float(value), float(value))
    else:
        first, second = value
        pair = (float(first), float(second))
    if not all(math.isfinite(item) and item >= 0.0 for item in pair):
        msg = f"{name} values must be non-negative and finite"
        raise ValueError(msg)
    return pair


class ElasticTransform(Transform):
    """Transform outlines with a smooth random displacement field.

    ``alpha`` controls displacement magnitude and ``sigma`` controls
    smoothness. Both are measured in em units and may be specified separately
    for the x and y axes.
    """

    _GRID_SIZE = 64
    _CANVAS_SIZE = 1.5

    def __init__(
        self,
        alpha: float | tuple[float, float] = 0.05,
        sigma: float | tuple[float, float] = 0.1,
    ) -> None:
        super().__init__()
        self.alpha = _pair(alpha, "alpha")
        self.sigma = _pair(sigma, "sigma")

    @classmethod
    def _smooth(cls, noise: torch.Tensor, sigma: float) -> torch.Tensor:
        if sigma == 0.0:
            return noise
        sigma_pixels = sigma * (cls._GRID_SIZE - 1) / cls._CANVAS_SIZE
        kernel_size = int(8.0 * sigma_pixels + 1.0)
        if kernel_size % 2 == 0:
            kernel_size += 1
        radius = kernel_size // 2
        positions = torch.arange(-radius, radius + 1, dtype=noise.dtype)
        kernel = torch.exp(-(positions**2) / (2.0 * sigma_pixels**2))
        kernel /= kernel.sum()
        noise = nn_functional.conv2d(
            noise, kernel.reshape(1, 1, 1, -1), padding=(0, radius)
        )
        return nn_functional.conv2d(
            noise, kernel.reshape(1, 1, -1, 1), padding=(radius, 0)
        )

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        fields = []
        for alpha, sigma in zip(self.alpha, self.sigma, strict=True):
            noise = torch.rand(1, 1, self._GRID_SIZE, self._GRID_SIZE) * 2.0 - 1.0
            fields.append(self._smooth(noise, sigma) * alpha)
        displacement = torch.cat(fields, dim=1).permute(0, 2, 3, 1)
        return {"displacement": displacement}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.elastic(inpt, params["displacement"])


__all__ = [
    "Affine",
    "ElasticTransform",
    "GaussianNoise",
    "HorizontalFlip",
    "RandomAffine",
    "RandomHorizontalFlip",
    "RandomRotation",
    "RandomScale",
    "RandomVerticalFlip",
    "VerticalFlip",
]
