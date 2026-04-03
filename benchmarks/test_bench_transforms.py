"""Benchmarks for torchfont.transforms operations.

Run locally with pytest-benchmark::

    pytest benchmarks/test_bench_transforms.py --benchmark-only

Run with asv::

    asv run --bench TransformQuadToCubic
    asv run --bench TransformLimitSequenceLength
    asv run --bench TransformPatchify
    asv run --bench TransformCompose
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmarks._helpers import make_glyph_sample
from torchfont.transforms import Compose, LimitSequenceLength, Patchify, QuadToCubic

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture

    from torchfont.datasets import GlyphSample


# ---------------------------------------------------------------------------
# pytest-benchmark tests
# ---------------------------------------------------------------------------


def test_bench_quad_to_cubic(
    benchmark: BenchmarkFixture,
    glyph_sample: GlyphSample,
) -> None:
    """Benchmark QuadToCubic on a 256-command sequence."""
    transform = QuadToCubic()
    benchmark(transform, glyph_sample)


def test_bench_limit_sequence_length(
    benchmark: BenchmarkFixture,
    glyph_sample: GlyphSample,
) -> None:
    """Benchmark LimitSequenceLength truncating a 256-command sequence to 128."""
    transform = LimitSequenceLength(max_len=128)
    benchmark(transform, glyph_sample)


def test_bench_patchify(
    benchmark: BenchmarkFixture,
    glyph_sample: GlyphSample,
) -> None:
    """Benchmark Patchify splitting a 256-command sequence into patches of 32."""
    transform = Patchify(patch_size=32)
    benchmark(transform, glyph_sample)


def test_bench_compose(
    benchmark: BenchmarkFixture,
    glyph_sample: GlyphSample,
) -> None:
    """Benchmark a full Compose pipeline: QuadToCubic, LimitSequenceLength, Patchify."""
    transform = Compose(
        [QuadToCubic(), LimitSequenceLength(max_len=128), Patchify(patch_size=32)]
    )
    benchmark(transform, glyph_sample)


# ---------------------------------------------------------------------------
# asv benchmark classes
# (discovered by `asv run`; ignored by pytest)
# ---------------------------------------------------------------------------


class TransformQuadToCubic:
    """asv benchmark for QuadToCubic."""

    params = [64, 256, 1024]
    param_names = ["n_commands"]

    def setup(self, n_commands: int) -> None:
        self._transform = QuadToCubic()
        self._sample = make_glyph_sample(n_commands)

    def time_transform(self, n_commands: int) -> None:
        self._transform(self._sample)


class TransformLimitSequenceLength:
    """asv benchmark for LimitSequenceLength."""

    params = [64, 256, 1024]
    param_names = ["n_commands"]

    def setup(self, n_commands: int) -> None:
        self._transform = LimitSequenceLength(max_len=n_commands // 2)
        self._sample = make_glyph_sample(n_commands)

    def time_transform(self, n_commands: int) -> None:
        self._transform(self._sample)


class TransformPatchify:
    """asv benchmark for Patchify."""

    params = [64, 256, 1024]
    param_names = ["n_commands"]

    def setup(self, n_commands: int) -> None:
        self._transform = Patchify(patch_size=32)
        self._sample = make_glyph_sample(n_commands)

    def time_transform(self, n_commands: int) -> None:
        self._transform(self._sample)


class TransformCompose:
    """asv benchmark for a full Compose pipeline."""

    params = [64, 256, 1024]
    param_names = ["n_commands"]

    def setup(self, n_commands: int) -> None:
        self._transform = Compose(
            [
                QuadToCubic(),
                LimitSequenceLength(max_len=n_commands // 2),
                Patchify(patch_size=32),
            ]
        )
        self._sample = make_glyph_sample(n_commands)

    def time_transform(self, n_commands: int) -> None:
        self._transform(self._sample)
