from __future__ import annotations

import torch

from torchfont.datasets import GlyphSample
from torchfont.io import CommandType


def make_glyph_sample(n: int = 256) -> GlyphSample:
    """Return a GlyphSample with *n* commands for use in benchmarks."""
    types = torch.full((n,), CommandType.LINE_TO.value, dtype=torch.long)
    types[0] = CommandType.MOVE_TO.value
    types[1::3] = CommandType.QUAD_TO.value
    types[-1] = CommandType.END.value
    coords = torch.randn(n, 6)
    return GlyphSample(types=types, coords=coords, style_idx=0, content_idx=0)
