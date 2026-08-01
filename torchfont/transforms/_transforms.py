"""Built-in transforms."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

import torch

from torchfont import _torchfont
from torchfont.datasets import (
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
)
from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import (
    Bitmap,
    GlyphData,
    Outline,
    OutlinePatches,
    Transform,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torchfont.transforms.bitmap import BitmapMode, FillRule


class LoadGlyph(Transform):
    """Load fixed-location glyph samples or references as semantic glyph data."""

    _transformed_types = (GlyphSample, GlyphRef)

    def transform(
        self, inpt: GlyphSample | GlyphRef, params: dict[str, Any]
    ) -> GlyphData | Outline:
        """Load one fixed-location glyph."""
        del params
        ref = inpt.ref if isinstance(inpt, GlyphSample) else inpt
        outline = _functional.load_glyph(ref)
        return (
            GlyphData(outline, inpt, ref.location)
            if isinstance(inpt, GlyphSample)
            else outline
        )


class RandomLocation(Transform):
    """Sample a variation location and load variable glyph samples or references."""

    _transformed_types = (VariableGlyphSample, VariableGlyphRef)

    def check_inputs(self, flat_inputs: list[object]) -> None:
        """Require one variable glyph because locations are sample-specific."""
        selected = [
            item
            for item, needs_transform in zip(
                flat_inputs, self._needs_transform_list(flat_inputs), strict=True
            )
            if needs_transform
        ]
        if len(selected) != 1:
            msg = "RandomLocation requires exactly one variable glyph input"
            raise ValueError(msg)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample one location for the selected variable glyph."""
        inpt = flat_inputs[0]
        ref = inpt.ref if isinstance(inpt, VariableGlyphSample) else inpt
        location: dict[str, float] = {}
        for tag, minimum, _default, maximum in _torchfont.variation_axes(
            ref.font.path, ref.font.ttc_index
        ):
            location[str(tag)] = (
                float(minimum)
                + (float(maximum) - float(minimum)) * torch.rand(()).item()
            )
        return {"location": location}

    def transform(
        self,
        inpt: VariableGlyphSample | VariableGlyphRef,
        params: dict[str, Any],
    ) -> GlyphData | Outline:
        """Load one variable glyph at its sampled location."""
        ref = inpt.ref if isinstance(inpt, VariableGlyphSample) else inpt
        location = params["location"]
        outline = _functional.load_glyph(ref, location)
        return (
            GlyphData(outline, inpt, location)
            if isinstance(inpt, VariableGlyphSample)
            else outline
        )


class _SimpleTransform(Transform):
    function: Callable[[Outline], Outline]

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return self.function(inpt)


class QuadToCubic(Transform):
    """Convert quadratic segments to cubic segments."""

    def __init__(self, *, merge_curves: bool = False) -> None:
        super().__init__()
        self.merge_curves = merge_curves

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        del params
        return _functional.quad_to_cubic(inpt, merge_curves=self.merge_curves)


class CubicToQuad(_SimpleTransform):
    """Convert cubic segments to quadratic segments."""

    function = staticmethod(_functional.cubic_to_quad)


class MergeCurves(_SimpleTransform):
    """Merge adjacent pieces of the same parent segment."""

    function = staticmethod(_functional.merge_curves)


class RemoveOverlaps(_SimpleTransform):
    """Merge overlapping subpaths."""

    function = staticmethod(_functional.remove_overlaps)


class NormalizeSubpathStartPoints(_SimpleTransform):
    """Choose a deterministic start point for each closed subpath."""

    function = staticmethod(_functional.normalize_subpath_start_points)


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
    """Flip all outlines in an input horizontally with probability ``p``."""

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
    """Flip all outlines in an input vertically with probability ``p``."""

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
        bounds = (float(value[0]), float(value[1]))
    if not all(math.isfinite(item) for item in bounds) or bounds[0] > bounds[1]:
        msg = "range values must be finite and ordered"
        raise ValueError(msg)
    return bounds


class RandomAffine(Transform):
    """Apply one randomly sampled affine transform to all outlines in an input."""

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
        if translate is not None and not all(
            math.isfinite(value) and value >= 0.0 for value in translate
        ):
            msg = "translate values must be non-negative and finite"
            raise ValueError(msg)
        if scale is not None and not all(
            math.isfinite(value) and value > 0.0 for value in scale
        ):
            msg = "scale values must be positive and finite"
            raise ValueError(msg)
        if scale is not None and scale[0] > scale[1]:
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
    """Add shared Gaussian coordinate noise to corresponding outline elements."""

    def __init__(self, std: float) -> None:
        super().__init__()
        if not math.isfinite(std):
            msg = "std must be finite"
            raise ValueError(msg)
        self.std = std

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.coords.size(0) for inpt in flat_inputs), default=0)
        return {"noise": torch.randn((length, 3, 2)) * self.std}

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.coord_jitter(inpt, params["noise"])


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
        return _functional.split_segments(
            inpt,
            values[0],
            values[1],
            split_probability=self.split_probability,
            split_range=self.split_range,
        )


class _RandomValuesTransform(Transform):
    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
        length = max((inpt.types.size(0) for inpt in flat_inputs), default=0)
        return {"values": torch.rand(length)}


class RandomRemoveOverlaps(_RandomValuesTransform):
    """Randomly simplify bbox-connected overlap groups."""

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.remove_overlap_groups(inpt, params["values"])


class RandomizeSubpathStartPoints(_RandomValuesTransform):
    """Choose a random start point for every closed subpath."""

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.set_subpath_start_points(inpt, params["values"])


class RandomizeSubpathOrder(_RandomValuesTransform):
    """Randomly permute whole subpaths."""

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Outline:
        return _functional.reorder_subpaths(inpt, params["values"])


class Patchify(Transform):
    """Split outlines into fixed-length patches."""

    def __init__(self, patch_size: int) -> None:
        super().__init__()
        self.patch_size = patch_size

    def transform(self, inpt: Outline, params: dict[str, Any]) -> OutlinePatches:
        del params
        return _functional.patchify(inpt, self.patch_size)


class RenderBitmap(Transform):
    """Render outlines into greyscale bitmap tensors."""

    def __init__(
        self,
        size: int = 64,
        mode: BitmapMode = "bbox_square",
        fill_rule: FillRule = "winding",
    ) -> None:
        super().__init__()
        self.size = size
        self.mode = mode
        self.fill_rule = fill_rule

    def transform(self, inpt: Outline, params: dict[str, Any]) -> Bitmap:
        del params
        return _functional.render_bitmap(inpt, self.size, self.mode, self.fill_rule)
