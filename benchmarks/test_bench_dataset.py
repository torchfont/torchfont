"""Benchmarks for GlyphDataset initialization and iteration.

Run locally with pytest-benchmark::

    pytest benchmarks/test_bench_dataset.py --benchmark-only

Run with asv::

    asv run --bench DatasetInit
    asv run --bench DatasetIter
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import TYPE_CHECKING

from benchmarks._helpers import BENCH_FONT_PATTERNS, copy_font_copies, fonts_dir
from torchfont.datasets import GlyphDataset

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

# Codepoints kept small so benchmarks focus on I/O and indexing, not rendering
_CODEPOINTS = tuple(range(0x41, 0x5B)) + tuple(range(0x61, 0x7B))  # A-Z and a-z


# ---------------------------------------------------------------------------
# pytest-benchmark tests
# ---------------------------------------------------------------------------


def test_bench_dataset_init_small(benchmark: BenchmarkFixture) -> None:
    """Benchmark GlyphDataset construction from a single font file."""
    benchmark(
        GlyphDataset,
        root=fonts_dir(),
        patterns=(BENCH_FONT_PATTERNS[0],),
        codepoints=_CODEPOINTS,
    )


def test_bench_dataset_init_large(
    benchmark: BenchmarkFixture,
    font_copies_dir: Path,
) -> None:
    """Benchmark GlyphDataset construction from a pseudo-large font collection."""
    benchmark(
        GlyphDataset,
        root=font_copies_dir,
        patterns=("**/*.ttf",),
        codepoints=_CODEPOINTS,
    )


def test_bench_dataset_iter_small(benchmark: BenchmarkFixture) -> None:
    """Benchmark a full iteration pass over a small dataset."""
    dataset = GlyphDataset(
        root=fonts_dir(),
        patterns=(BENCH_FONT_PATTERNS[0],),
        codepoints=_CODEPOINTS,
    )

    def _iterate() -> None:
        for _ in dataset:
            pass

    benchmark(_iterate)


def test_bench_dataset_iter_large(
    benchmark: BenchmarkFixture,
    font_copies_dir: Path,
) -> None:
    """Benchmark a full iteration pass over a pseudo-large dataset."""
    dataset = GlyphDataset(
        root=font_copies_dir,
        patterns=("**/*.ttf",),
        codepoints=_CODEPOINTS,
    )

    def _iterate() -> None:
        for _ in dataset:
            pass

    benchmark(_iterate)


# ---------------------------------------------------------------------------
# asv benchmark classes
# (discovered by `asv run`; ignored by pytest)
# ---------------------------------------------------------------------------


class DatasetInit:
    """asv benchmark for GlyphDataset construction."""

    params = [1, 50]
    param_names = ["n_copies"]

    def setup(self, n_copies: int) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        copy_font_copies(root, n_copies)
        self._root = root

    def teardown(self, n_copies: int) -> None:
        self._tmpdir.cleanup()

    def time_init(self, n_copies: int) -> None:
        GlyphDataset(
            root=self._root,
            patterns=("**/*.ttf",),
            codepoints=_CODEPOINTS,
        )


class DatasetIter:
    """asv benchmark for iterating through an entire GlyphDataset."""

    params = [1, 50]
    param_names = ["n_copies"]

    def setup(self, n_copies: int) -> None:
        self._tmpdir = tempfile.TemporaryDirectory()
        root = Path(self._tmpdir.name)
        copy_font_copies(root, n_copies)
        self._dataset = GlyphDataset(
            root=root,
            patterns=("**/*.ttf",),
            codepoints=_CODEPOINTS,
        )

    def teardown(self, n_copies: int) -> None:
        self._tmpdir.cleanup()

    def time_iter(self, n_copies: int) -> None:
        for _ in self._dataset:
            pass
