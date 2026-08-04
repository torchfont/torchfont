import shutil
from pathlib import Path

BENCH_FONT_PATTERNS = (
    "source-sans/SourceSans3-Regular.ttf",
    "source-sans/SourceSans3-Regular.otf",
    "source-serif/SourceSerif4Variable-Roman.ttf",
)


def fonts_dir() -> Path:
    return Path(__file__).parent.parent / "fonts"


def copy_font_copies(root: Path, n_copies: int) -> None:
    """Copy benchmark fonts into *root* with *n_copies* copies per font."""
    source_root = fonts_dir()
    for pattern in BENCH_FONT_PATTERNS:
        src = source_root / pattern
        for i in range(n_copies):
            dst = root / str(i) / pattern
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)
