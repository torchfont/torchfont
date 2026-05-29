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
MAX_FAILURE_RATE = 0.001  # 0.1 %

# Outer rectangle that surrounds all Google Fonts glyphs.
# Measured via tight_bbox over the full dataset: x ∈ [-3.24, 12.26], y ∈ [-2.96, 2.56].
# Adding ~0.75 margin and rounding to clean values gives these bounds.
# The path is clockwise in y-up coordinates, which maps to -1 winding after the y-flip
# applied by the renderer, shifting every pixel's winding number by -1 (P - H).
# With this, N0 xor E0 catches w = even non-zero, N1 xor E1 catches w = odd |w| > 1,
# together covering all w ∉ {0, 1}.
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
# Clockwise in y-up: bottom-left → top-left → top-right → bottom-right → close
_OUTER_RECT_COORDS = torch.tensor(
    [
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MIN, _RECT_Y_MIN],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MIN, _RECT_Y_MAX],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MAX, _RECT_Y_MAX],
        [0.0, 0.0, 0.0, 0.0, _RECT_X_MAX, _RECT_Y_MIN],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=torch.float32,
)


def _hard_diff(a: Tensor, b: Tensor) -> Tensor:
    return ((a == 255) & (b == 0)) | ((a == 0) & (b == 255))


def _transform(sample: GlyphSample) -> Tensor:
    tf_types, tf_coords = remove_overlaps(sample.types, sample.coords)

    aug_types = torch.cat([tf_types, _OUTER_RECT_TYPES])
    aug_coords = torch.cat([tf_coords, _OUTER_RECT_COORDS])

    orig_winding = render_bitmap(
        sample.types, sample.coords, size=64, mode="fixed", fill_rule="winding"
    )
    tf_winding = render_bitmap(
        tf_types, tf_coords, size=64, mode="fixed", fill_rule="winding"
    )
    tf_even_odd = render_bitmap(
        tf_types, tf_coords, size=64, mode="fixed", fill_rule="even_odd"
    )
    tf_winding_aug = render_bitmap(
        aug_types, aug_coords, size=64, mode="fixed", fill_rule="winding"
    )
    tf_even_odd_aug = render_bitmap(
        aug_types, aug_coords, size=64, mode="fixed", fill_rule="even_odd"
    )

    bitmap_mismatch = _hard_diff(orig_winding, tf_winding).any()

    # Overlap detection: N0 xor E0 catches w = even non-zero (same-direction overlaps);
    # N1 xor E1 catches w = odd |w| > 1 (wrong-direction contours).
    # Together they cover all w ∉ {0, 1}.
    tf_has_overlaps = (
        _hard_diff(tf_winding, tf_even_odd)
        | _hard_diff(tf_winding_aug, tf_even_odd_aug)
    ).any()

    failed = bitmap_mismatch | tf_has_overlaps
    if failed:
        reasons = []
        if bitmap_mismatch:
            reasons.append("bitmap_mismatch")
        if tf_has_overlaps:
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
