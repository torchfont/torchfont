from __future__ import annotations

from typing import Any, ClassVar, Literal

import torch

from torchfont import _torchfont
from torchfont._glyph import (
    CodepointData,
    CodepointSample,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
    _registered_axis_targets,
)
from torchfont._ops import _restore
from torchfont._outline import Outline
from torchfont.transforms._transform import Transform


class LoadGlyph(Transform):
    """Load referenced glyphs at default or randomly sampled locations."""

    _transformed_types: ClassVar[tuple[type[Any], ...]] = (
        CodepointSample,
        GlyphIdSample,
        GlyphRef,
    )

    def __init__(self, location: Literal["default", "random"] = "default") -> None:
        super().__init__()
        if location not in ("default", "random"):
            msg = "location must be 'default' or 'random'"
            raise ValueError(msg)
        self.location = location

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        """Sample shared random parameters for all inputs."""
        if self.location == "default":
            return {}
        seed = torch.empty((), dtype=torch.int64).random_().item()
        return {"seed": seed}

    def transform(
        self,
        inpt: CodepointSample | GlyphIdSample | GlyphRef,
        params: dict[str, Any],
    ) -> CodepointData | GlyphIdData | Outline:
        """Load one referenced glyph using the shared parameters."""
        ref = inpt if isinstance(inpt, GlyphRef) else inpt.ref
        requested_location = (
            None
            if self.location == "default"
            else _random_location(ref, params["seed"])
        )
        (raw_types, raw_coords), location_items, axis_values = _torchfont.load_glyph(
            ref.font.path,
            ref.font.face_index,
            ref.glyph_id,
            requested_location,
        )
        outline = Outline(*_restore(raw_types, raw_coords))
        location = dict(location_items)
        if isinstance(inpt, GlyphRef):
            return outline
        if isinstance(inpt, GlyphIdSample):
            return GlyphIdData(
                data=outline,
                ref=ref,
                location=location,
                font_idx=inpt.font_idx,
                **_registered_axis_targets(axis_values),
            )
        return CodepointData(
            data=outline,
            ref=ref,
            location=location,
            codepoint=inpt.codepoint,
            font_idx=inpt.font_idx,
            character_idx=inpt.character_idx,
            **_registered_axis_targets(axis_values),
        )

    def extra_repr(self) -> str:
        return f"location={self.location}"


def _random_location(ref: GlyphRef, seed: int) -> dict[str, float]:
    generator = torch.Generator().manual_seed(seed)
    location: dict[str, float] = {}
    for tag, minimum, _default, maximum in _torchfont.variation_axes(
        ref.font.path, ref.font.face_index
    ):
        location[str(tag)] = (
            torch.empty(()).uniform_(minimum, maximum, generator=generator).item()
        )
    return location


__all__ = ["LoadGlyph"]
