import logging
from pathlib import Path

import pytest
import torch
from torch import Tensor
from torch.utils.data import DataLoader

from torchfont import GlyphIdSample
from torchfont import transforms as T  # noqa: N812
from torchfont.datasets import GlyphIdDataset
from torchfont.transforms import functional as F  # noqa: N812

logger = logging.getLogger(__name__)

GOOGLE_FONTS_ROOT = Path("data/google/fonts")


def _augmentation() -> T.Compose:
    return T.Compose(
        [
            T.LoadGlyph(location="random"),
            T.QuadToCubic(merge_curves=True),
            T.RandomApply(T.RandomSubpathDropout(p=0.1), p=0.3),
            T.RandomApply(
                T.RandomChoice(
                    [
                        T.Compose(
                            [T.RemoveOverlaps(), T.QuadToCubic(merge_curves=False)]
                        ),
                        T.Compose(
                            [
                                T.RandomRemoveOverlaps(),
                                T.QuadToCubic(merge_curves=False),
                            ]
                        ),
                    ]
                ),
                p=2 / 3,
            ),
            T.RandomApply(T.RandomScale(scale_x=(0.8, 1.2), scale_y=(0.8, 1.2))),
            T.RandomHorizontalFlip(),
            T.RandomVerticalFlip(),
            T.RandomApply(
                T.RandomAffine(degrees=180.0, translate=(0.05, 0.05), shear=10.0),
                p=0.8,
            ),
            T.RandomApply(T.ElasticTransform(alpha=0.08, sigma=0.12), p=0.3),
            T.RandomApply(T.RandomSplitSegments(split_probability=0.3), p=0.3),
            T.RandomSubpathStartPoints(),
            T.RandomSubpathOrder(),
            T.RandomApply(T.GaussianNoise(sigma=0.005), p=0.3),
        ]
    )


class _CheckWinding:
    def __init__(self) -> None:
        self.augment = _augmentation()
        self.reverse = T.RandomWinding()

    def __call__(self, sample: GlyphIdSample) -> Tensor:
        torch.manual_seed(sample.ref.glyph_id * 1009 + sample.font_idx)
        outline = self.augment(sample.ref)
        results = (
            self.reverse(outline),
            F.normalize_winding(outline),
            F.normalize_winding(outline, clockwise=False),
        )
        mismatch = False
        for fill_rule in ("winding", "even_odd"):
            original = F.render_bitmap(outline, 128, fill_rule=fill_rule)
            for result in results:
                rendered = F.render_bitmap(result, 128, fill_rule=fill_rule)
                hard_diff = ((original == 255) & (rendered == 0)) | (
                    (original == 0) & (rendered == 255)
                )
                mismatch |= bool(hard_diff.any())
        if mismatch:
            logger.warning(
                "winding bitmap mismatch: %s gid %s",
                sample.ref.font.path,
                sample.ref.glyph_id,
            )
        return torch.tensor(mismatch)


@pytest.mark.google_fonts
def test_augmented_winding_google_fonts(request: pytest.FixtureRequest) -> None:
    if not GOOGLE_FONTS_ROOT.is_dir():
        pytest.fail(f"Google Fonts checkout not available: {GOOGLE_FONTS_ROOT}")
    limit: int | None = request.config.getoption("--limit")
    dataset = GlyphIdDataset(
        GOOGLE_FONTS_ROOT,
        max_length=192,
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
        transform=_CheckWinding(),
    )
    loader = DataLoader(
        dataset,
        batch_size=128,
        shuffle=True,
        num_workers=8,
        generator=torch.Generator().manual_seed(20261010),
    )
    total = 0
    mismatches = 0
    for batch in loader:
        total += batch.numel()
        mismatches += int(batch.sum())
        if total % 10240 == 0:
            logger.warning(
                "winding progress: %s glyphs, %s mismatches", total, mismatches
            )
        if limit is not None and total >= limit:
            break
    assert mismatches == 0, f"winding changed coverage for {mismatches}/{total} glyphs"
