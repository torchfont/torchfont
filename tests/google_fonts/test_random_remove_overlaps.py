import logging
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import ElementType, GlyphSample, Outline
from torchfont.transforms import RandomRemoveOverlaps
from torchfont.transforms import functional as _functional

logger = logging.getLogger(__name__)

GOOGLE_FONTS_ROOT = Path("data/google/fonts")

# Skia PathOps has known edge-case bugs; allow up to this fraction of glyphs to fail.
MAX_FAILURE_RATE = 0.001  # 0.1 %
MIN_CHANGE_RATE = 0.1  # 10 %
BITMAP_SIZE = 128


def _hard_diff(a: Tensor, b: Tensor) -> Tensor:
    return ((a == 255) & (b == 0)) | ((a == 0) & (b == 255))


def _transform(sample: GlyphSample) -> Tensor:
    outline = _functional.load_glyph(sample.ref)
    types, coords = outline.types, outline.coords
    torch.manual_seed(sample.ref.codepoint)
    simplified = RandomRemoveOverlaps()(Outline(types, coords))
    simplified_types, simplified_coords = simplified.types, simplified.coords

    original = _functional.render_bitmap(
        Outline(types, coords),
        size=BITMAP_SIZE,
        mode="fixed",
        fill_rule="winding",
    )
    simplified_bitmap = _functional.render_bitmap(
        simplified,
        size=BITMAP_SIZE,
        mode="fixed",
        fill_rule="winding",
    )

    failed = _hard_diff(original, simplified_bitmap).any()
    if failed:
        logger.warning(
            "random_remove_overlaps bitmap mismatch: %s U+%04X %s",
            sample.ref.font.path,
            sample.ref.codepoint,
            sample.ref.location,
        )
    changed = not (
        torch.equal(types, simplified_types) and torch.equal(coords, simplified_coords)
    )
    subpath_reduction = max(
        0,
        types.tolist().count(ElementType.MOVE_TO.value)
        - simplified_types.tolist().count(ElementType.MOVE_TO.value),
    )
    return torch.tensor([failed, changed, subpath_reduction], dtype=torch.long)


@pytest.mark.google_fonts
def test_random_remove_overlaps_google_fonts(
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
    changed = 0
    subpath_reduction = 0
    next_progress = 10_000
    for batch in dataloader:
        failures += batch[:, 0].sum().item()
        changed += batch[:, 1].sum().item()
        subpath_reduction += batch[:, 2].sum().item()
        total += batch.size(0)
        if total >= next_progress:
            logger.warning(
                "random_remove_overlaps progress: %s glyphs, %s failures, "
                "%s changed, %s subpaths removed",
                f"{total:,}",
                f"{failures:,}",
                f"{changed:,}",
                f"{subpath_reduction:,}",
            )
            next_progress = total + 10_000
        if limit is not None and total >= limit:
            break

    failure_rate = failures / max(1, total)
    change_rate = changed / max(1, total)
    logger.warning(
        "random_remove_overlaps result: %s/%s changed (%.4f%%), %s subpaths removed",
        f"{changed:,}",
        f"{total:,}",
        change_rate * 100,
        f"{subpath_reduction:,}",
    )
    assert failure_rate <= MAX_FAILURE_RATE, (
        f"random_remove_overlaps failure rate {failure_rate:.4%} "
        f"({failures}/{total}) exceeds threshold {MAX_FAILURE_RATE:.4%}"
    )
    assert change_rate >= MIN_CHANGE_RATE, (
        f"random_remove_overlaps change rate {change_rate:.4%} "
        f"({changed}/{total}) is below threshold {MIN_CHANGE_RATE:.4%}"
    )
