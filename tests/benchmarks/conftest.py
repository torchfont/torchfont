from pathlib import Path

import pytest

from tests.benchmarks._helpers import copy_font_copies

# Number of per-font copies; 50 copies x 3 fonts = 150 font files
_BENCH_COPIES = 50


@pytest.fixture(scope="session")
def font_copies_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Temporary directory containing *_BENCH_COPIES* duplicates of each font.

    Simulates a pseudo-large dataset from a small number of source fonts, as
    recommended by the benchmarking strategy in the project issue.
    """
    root = tmp_path_factory.mktemp("bench_fonts")
    copy_font_copies(root, _BENCH_COPIES)
    return root
