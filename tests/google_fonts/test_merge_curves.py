import logging
from pathlib import Path

import pytest
from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import merge_curves, render_bitmap

logger = logging.getLogger(__name__)

GOOGLE_FONTS_ROOT = Path("data/google/fonts")

# Allow rare rasterizer edge cases while still catching corpus-wide regressions.
MAX_FAILURE_RATE = 0.001  # 0.1 %
BITMAP_SIZE = 128


def _hard_diff(a: Tensor, b: Tensor) -> Tensor:
    """Ignore antialiasing noise and detect foreground/background changes."""
    return ((a == 255) & (b == 0)) | ((a == 0) & (b == 255))


def _transform(sample: GlyphSample) -> Tensor:
    merged_types, merged_coords = merge_curves(sample.types, sample.coords)

    original = render_bitmap(
        sample.types,
        sample.coords,
        size=BITMAP_SIZE,
        mode="fixed",
        fill_rule="winding",
    )
    merged = render_bitmap(
        merged_types,
        merged_coords,
        size=BITMAP_SIZE,
        mode="fixed",
        fill_rule="winding",
    )

    failed = _hard_diff(original, merged).any()
    if failed:
        logger.warning(
            "merge_curves bitmap mismatch: %s U+%04X %s",
            sample.name.family_name,
            sample.codepoint,
            sample.glyph_name,
        )
    return failed


@pytest.mark.google_fonts
def test_merge_curves_google_fonts(
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
        f"merge_curves failure rate {failure_rate:.4%} ({failures}/{total}) "
        f"exceeds threshold {MAX_FAILURE_RATE:.4%}"
    )
