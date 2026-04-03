from __future__ import annotations

import shutil
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from benchmarks._helpers import make_glyph_sample

if TYPE_CHECKING:
    from torchfont.datasets import GlyphSample

FONTS_DIR = Path(__file__).parent.parent / "tests" / "fonts"

# Source fonts used to build pseudo-large datasets
_BENCH_FONT_PATTERNS = (
    "lato/Lato-Regular.ttf",
    "ubuntu/Ubuntu-Regular.ttf",
    "ptsans/PT_Sans-Web-Regular.ttf",
)

# Number of per-font copies; 50 copies x 3 fonts = 150 font files
_BENCH_COPIES = 50


@pytest.fixture(scope="session")
def font_copies_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory containing *_BENCH_COPIES* duplicates of each font.

    Simulates a pseudo-large dataset from a small number of source fonts, as
    recommended by the benchmarking strategy in the project issue.
    """
    root = tmp_path_factory.mktemp("bench_fonts")
    for pattern in _BENCH_FONT_PATTERNS:
        src = FONTS_DIR / pattern
        for i in range(_BENCH_COPIES):
            dst = root / str(i) / pattern
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
    return root


@pytest.fixture(scope="module")
def glyph_sample() -> GlyphSample:
    """A realistic-length GlyphSample for transform benchmarks."""
    return make_glyph_sample(256)
