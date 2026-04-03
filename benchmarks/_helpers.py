from __future__ import annotations

import shutil
from pathlib import Path

BENCH_FONT_PATTERNS = (
    "lato/Lato-Regular.ttf",
    "ubuntu/Ubuntu-Regular.ttf",
    "ptsans/PT_Sans-Web-Regular.ttf",
)


def fonts_dir() -> Path:
    return Path(__file__).parent.parent / "tests" / "fonts"


def copy_font_copies(root: Path, n_copies: int) -> None:
    """Copy benchmark fonts into *root* with *n_copies* copies per font."""
    source_root = fonts_dir()
    for pattern in BENCH_FONT_PATTERNS:
        src = source_root / pattern
        for i in range(n_copies):
            dst = root / str(i) / pattern
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
