"""Glyph loading transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torchfont import _torchfont
from torchfont.structures import (
    GlyphData,
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
)
from torchfont.transforms import functional as _functional
from torchfont.transforms._transform import Transform

if TYPE_CHECKING:
    from torchfont.structures import Outline


class LoadGlyph(Transform):
    """Load fixed-location glyph samples or references as semantic glyph data."""

    _transformed_types = (GlyphSample, GlyphRef)

    def transform(
        self, inpt: GlyphSample | GlyphRef, params: dict[str, Any]
    ) -> GlyphData | Outline:
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

    def check_same_params(self, selected_inputs: list[object]) -> None:
        refs: list[VariableGlyphRef] = []
        for item in selected_inputs:
            if isinstance(item, VariableGlyphSample):
                refs.append(item.ref)
            elif isinstance(item, VariableGlyphRef):
                refs.append(item)
        if refs and any(ref.font != refs[0].font for ref in refs[1:]):
            msg = "RandomLocation can share parameters only within one font"
            raise ValueError(msg)

    def make_params(self, flat_inputs: list[Any]) -> dict[str, Any]:
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
        ref = inpt.ref if isinstance(inpt, VariableGlyphSample) else inpt
        location = params["location"]
        outline = _functional.load_glyph(ref, location)
        return (
            GlyphData(outline, inpt, location)
            if isinstance(inpt, VariableGlyphSample)
            else outline
        )


__all__ = ["LoadGlyph", "RandomLocation"]
