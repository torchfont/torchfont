import logging
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.io import ElementType
from torchfont.transforms import remove_overlaps, render_bitmap

logger = logging.getLogger(__name__)

GOOGLE_FONTS_ROOT = Path("data/google/fonts")

# Skia PathOps has known edge-case bugs; allow up to this fraction of glyphs to fail.
MAX_FAILURE_RATE = 0.01  # 1 %

# Outer rectangle covering all Google Fonts glyphs with margin.
# Prepending CW or CCW variants shifts every pixel's winding number w:
#   CW rect (y-up) → w-1;  CCW rect (y-up) → w+1.
# Winding renders of the shifted path (255 iff shifted w ≠ 0) then satisfy:
#   simplified     = 255 iff w ≠  0
#   simplified_cw  = 255 iff w ≠  1
#   simplified_ccw = 255 iff w ≠ -1
#   AND of all three = 255 iff w ∉ {-1, 0, 1}.
_RECT_X_MIN, _RECT_X_MAX = -4.0, 13.0
_RECT_Y_MIN, _RECT_Y_MAX = -4.0, 3.5

_OUTER_RECT_TYPES = torch.tensor(
    [
        ElementType.MOVE_TO,
        ElementType.LINE_TO,
        ElementType.LINE_TO,
        ElementType.LINE_TO,
        ElementType.CLOSE,
    ],
    dtype=torch.long,
)
# Clockwise in y-up
_OUTER_RECT_COORDS_CW = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MIN, _RECT_Y_MIN],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MIN, _RECT_Y_MAX],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MAX, _RECT_Y_MAX],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MAX, _RECT_Y_MIN],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)
# Counter-clockwise in y-up
_OUTER_RECT_COORDS_CCW = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MIN, _RECT_Y_MIN],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MAX, _RECT_Y_MIN],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MAX, _RECT_Y_MAX],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MIN, _RECT_Y_MAX],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def _hard_diff(a: Tensor, b: Tensor) -> Tensor:
    return ((a == 255) & (b == 0)) | ((a == 0) & (b == 255))


def _transform(sample: GlyphSample) -> Tensor:
    simplified_types, simplified_coords = remove_overlaps(sample.types, sample.coords)

    # Prepend outer rect before glyph contours; End token truncates if appended.
    prepended_types = torch.cat([_OUTER_RECT_TYPES, simplified_types])
    cw_coords = torch.cat([_OUTER_RECT_COORDS_CW, simplified_coords])
    ccw_coords = torch.cat([_OUTER_RECT_COORDS_CCW, simplified_coords])

    original = render_bitmap(
        sample.types, sample.coords, size=64, mode="fixed", fill_rule="winding"
    )
    simplified = render_bitmap(
        simplified_types, simplified_coords, size=64, mode="fixed", fill_rule="winding"
    )
    simplified_cw = render_bitmap(
        prepended_types, cw_coords, size=64, mode="fixed", fill_rule="winding"
    )
    simplified_ccw = render_bitmap(
        prepended_types, ccw_coords, size=64, mode="fixed", fill_rule="winding"
    )

    bitmap_mismatch = _hard_diff(original, simplified).any()
    has_overlaps = (
        (simplified == 255) & (simplified_cw == 255) & (simplified_ccw == 255)
    ).any()

    failed = bitmap_mismatch | has_overlaps
    if failed:
        reasons = []
        if bitmap_mismatch:
            reasons.append("bitmap_mismatch")
        if has_overlaps:
            reasons.append("has_overlaps")
        logger.warning(
            "remove_overlaps failure [%s]: %s U+%04X %s",
            ",".join(reasons),
            sample.name.family_name,
            sample.codepoint,
            sample.glyph_name,
        )
    return failed


@pytest.mark.google_fonts
def test_remove_overlaps_google_fonts(
    request: pytest.FixtureRequest,
) -> None:
    if not GOOGLE_FONTS_ROOT.is_dir():
        pytest.fail(f"Google Fonts checkout not available: {GOOGLE_FONTS_ROOT}")

    limit: int | None = request.config.getoption("--limit")

    dataset = GlyphDataset(
        root=GOOGLE_FONTS_ROOT,
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
            "!ofl/handjet/*.ttf",
            "!ofl/bitcount*/*.ttf",
        ),
        transform=_transform,
    )
    dataloader = DataLoader(
        dataset,
        batch_size=256,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
    )

    total = 0
    failures = 0
    for batch in dataloader:
        failures += batch.sum().item()
        total += batch.numel()
        if limit is not None and total >= limit:
            break

    failure_rate = failures / max(1, total)
    assert failure_rate <= MAX_FAILURE_RATE, (
        f"remove_overlaps failure rate {failure_rate:.4%} ({failures}/{total}) "
        f"exceeds threshold {MAX_FAILURE_RATE:.4%}"
    )
