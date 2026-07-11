"""Bridge from dataset glyph references to outline tensors."""

from __future__ import annotations

from typing import TYPE_CHECKING, overload

import torch

from torchfont import _torchfont
from torchfont.datasets import GlyphRef, VariableGlyphRef
from torchfont.io import COORD_DIM

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor


@overload
def load_glyph(ref: GlyphRef) -> tuple[Tensor, Tensor]: ...


@overload
def load_glyph(
    ref: VariableGlyphRef,
    location: Mapping[str, float] | None = None,
) -> tuple[Tensor, Tensor]: ...


def load_glyph(
    ref: GlyphRef | VariableGlyphRef,
    location: Mapping[str, float] | None = None,
) -> tuple[Tensor, Tensor]:
    """Load one glyph outline as ``(types, coords)`` tensors."""
    if isinstance(ref, GlyphRef):
        if location is not None:
            msg = "location cannot override a GlyphRef location"
            raise ValueError(msg)
        glyph_location: Mapping[str, float] | None = ref.location
    elif isinstance(ref, VariableGlyphRef):
        glyph_location = location
    else:
        msg = f"expected GlyphRef or VariableGlyphRef, got {type(ref).__name__}"
        raise TypeError(msg)

    raw_types, raw_coords = _torchfont.load_glyph(
        ref.font.path,
        ref.font.ttc_index,
        ref.codepoint,
        _location_arg(glyph_location),
    )
    types = torch.from_numpy(raw_types)
    coords = torch.from_numpy(raw_coords).view(-1, COORD_DIM)
    return types, coords


def _location_arg(location: Mapping[str, float] | None) -> dict[str, float] | None:
    if location is None:
        return None
    return {str(tag): float(value) for tag, value in location.items()}
