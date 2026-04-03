"""Composable transforms for polishing tensor glyph sequences before training.

Notes:
    Each transform is intentionally small and deterministic, allowing you to mix
    and match them in :class:`torch.utils.data.Dataset` pipelines without losing
    track of where normalization occurs.

Examples:
    Normalize and patch glyph tensors in a single pipeline::

        pipeline = Compose([QuadToCubic(), LimitSequenceLength(256), Patchify(32)])

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from torchfont.io import COORD_DIM, CommandType

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torchfont.datasets import GlyphSample


class Compose:
    """Apply a curated list of transform callables to every sample.

    See Also:
        LimitSequenceLength: Useful when a hard cap is required before adding
        more expensive stages.

    """

    def __init__(
        self,
        transforms: Sequence[Callable[[GlyphSample], GlyphSample]],
    ) -> None:
        """Store the ordered transform pipeline.

        Args:
            transforms (Sequence[Callable]): Operations that accept and return
                compatible sample types.
                Ordering matters, so place
                stateful or lossy transforms later in the list.

        Examples:
            Combine truncation with patching::

                Compose([LimitSequenceLength(256), Patchify(32)])

        """
        self.transforms = transforms

    def __call__(self, sample: GlyphSample) -> GlyphSample:
        """Apply every transform in order to the provided sample.

        Args:
            sample (GlyphSample): Input sample.

        Returns:
            GlyphSample: Resulting sample after all transformations are applied.

        Examples:
            Run the composed pipeline on a glyph sample::

                sample = pipeline(sample)

        """
        for t in self.transforms:
            sample = t(sample)
        return sample


class LimitSequenceLength:
    """Trim glyph sequences to a predictable maximum length.

    See Also:
        Patchify: Converts sequences into fixed-size blocks after truncation so
        transformer-style models see a uniform layout.

    """

    def __init__(self, max_len: int) -> None:
        """Initialize the transform with the desired maximum length.

        Args:
            max_len (int): Maximum number of time steps to keep. Must be
                non-negative. Any surplus command or coordinate pairs are
                discarded.

        Examples:
            Cap sequences at 512 steps::

                LimitSequenceLength(512)

        """
        if max_len < 0:
            msg = "max_len must be >= 0"
            raise ValueError(msg)
        self.max_len = max_len

    @staticmethod
    def _validate_sample(sample: GlyphSample) -> None:
        """Validate the untruncated sample contract required by the transform."""
        expected_types_ndim = 1
        expected_coords_ndim = 2

        if sample.types.ndim != expected_types_ndim:
            msg = (
                f"LimitSequenceLength expects types.ndim == "
                f"{expected_types_ndim}; got {sample.types.ndim}"
            )
            raise ValueError(msg)
        if sample.coords.ndim != expected_coords_ndim:
            msg = (
                f"LimitSequenceLength expects coords.ndim == "
                f"{expected_coords_ndim}; got {sample.coords.ndim}"
            )
            raise ValueError(msg)
        if sample.coords.shape[1] != COORD_DIM:
            msg = (
                f"LimitSequenceLength expects coords.shape[1] == {COORD_DIM}; "
                f"got {sample.coords.shape[1]}"
            )
            raise ValueError(msg)
        if sample.types.shape[0] != sample.coords.shape[0]:
            msg = (
                "LimitSequenceLength expects types.shape[0] and "
                "coords.shape[0] to match; "
                f"got {sample.types.shape[0]} and {sample.coords.shape[0]}"
            )
            raise ValueError(msg)

    def __call__(self, sample: GlyphSample) -> GlyphSample:
        """Clip the sequence and coordinate tensors to the specified length.

        Args:
            sample (GlyphSample): Input sample.

        Returns:
            GlyphSample: Sample with ``types`` and ``coords`` truncated to
            ``max_len`` steps.

        Warnings:
            Elements beyond ``max_len`` are removed rather than padded or
            aggregated, so downstream code should account for the shorter tail.

        Examples:
            Clamp a sample to 128 steps::

                sample = LimitSequenceLength(128)(sample)

        """
        self._validate_sample(sample)
        sample_type = type(sample)
        return sample_type(
            types=sample.types[: self.max_len],
            coords=sample.coords[: self.max_len],
            metrics=sample.metrics,
            style_idx=sample.style_idx,
            content_idx=sample.content_idx,
        )


class QuadToCubic:
    """Convert quadratic segments into cubic segments in tensor form.

    Notes:
        This transform rewrites only `CommandType.QUAD_TO` rows and leaves all
        other commands untouched. Coordinate dimensionality stays at 6.

    """

    def __call__(self, sample: GlyphSample) -> GlyphSample:
        """Convert `QUAD_TO` entries to `CURVE_TO`.

        Args:
            sample (GlyphSample): Input sample whose ``types`` values follow
                `CommandType` and whose ``coords`` last dimension is at least
                6 and layout `[cx0, cy0, cx1, cy1, x, y]`.

        Returns:
            GlyphSample: Sample with quadratic segments rewritten as cubic
            segments. Returns the original sample unchanged if no ``QUAD_TO``
            commands are present.

        """
        types = sample.types
        coords = sample.coords
        flat_types = types.reshape(-1)
        flat_coords = coords.reshape(-1, coords.size(-1))
        quad = flat_types == CommandType.QUAD_TO.value

        if not torch.any(quad):
            return sample

        out_types = flat_types.clone()
        out_coords = flat_coords.clone()

        # In valid outline streams, the previous command endpoint is the
        # current point for a quadratic segment.
        prev = torch.zeros_like(out_coords[:, 0:2])
        prev[1:] = out_coords[:-1, 4:6]

        q_prev = prev[quad]
        q_ctrl = out_coords[quad, 0:2]
        q_end = out_coords[quad, 4:6]

        out_coords[quad, 0:2] = q_prev + (2.0 / 3.0) * (q_ctrl - q_prev)
        out_coords[quad, 2:4] = q_end + (2.0 / 3.0) * (q_ctrl - q_end)
        out_types[quad] = CommandType.CURVE_TO.value

        sample_type = type(sample)
        return sample_type(
            types=out_types.view_as(types),
            coords=out_coords.view_as(coords),
            metrics=sample.metrics,
            style_idx=sample.style_idx,
            content_idx=sample.content_idx,
        )


class Patchify:
    """Pad glyph sequences and reshape them into uniform, model-friendly patches.

    See Also:
        LimitSequenceLength: Apply beforehand when you need a strict ceiling on
        the number of emitted patches.

    """

    def __init__(self, patch_size: int) -> None:
        """Configure the patch length for reshaping sequences.

        Args:
            patch_size (int): Number of time steps captured in each patch. Must
                be positive. Choose values that align with the receptive field
                of your downstream model.

        Examples:
            Create 32-step patches for transformer models::

                Patchify(32)

        """
        if patch_size < 1:
            msg = "patch_size must be >= 1"
            raise ValueError(msg)
        self.patch_size = patch_size

    @staticmethod
    def _validate_sample(sample: GlyphSample) -> None:
        """Validate the unpatchified sample contract required by Patchify."""
        expected_types_ndim = 1
        expected_coords_ndim = 2

        if sample.types.ndim != expected_types_ndim:
            msg = (
                f"Patchify expects types.ndim == {expected_types_ndim}; "
                f"got {sample.types.ndim}"
            )
            raise ValueError(msg)
        if sample.coords.ndim != expected_coords_ndim:
            msg = (
                f"Patchify expects coords.ndim == {expected_coords_ndim}; "
                f"got {sample.coords.ndim}"
            )
            raise ValueError(msg)
        if sample.coords.shape[1] != COORD_DIM:
            msg = (
                f"Patchify expects coords.shape[1] == {COORD_DIM}; "
                f"got {sample.coords.shape[1]}"
            )
            raise ValueError(msg)
        if sample.types.shape[0] != sample.coords.shape[0]:
            msg = (
                "Patchify expects types.shape[0] and coords.shape[0] to match; "
                f"got {sample.types.shape[0]} and {sample.coords.shape[0]}"
            )
            raise ValueError(msg)

    def __call__(self, sample: GlyphSample) -> GlyphSample:
        """Pad and reshape sequences into contiguous patches.

        Args:
            sample (GlyphSample): Input sample.

        Returns:
            GlyphSample: Sample whose ``types`` have shape
            ``(num_patches, patch_size)`` and ``coords`` have shape
            ``(num_patches, patch_size, 6)``. Trailing zeros are added
            only when needed for alignment.

        Tips:
            Pair with :class:`LimitSequenceLength` to bound the worst-case number
            of patches in a batch.

        Examples:
            Reshape a glyph sequence into patches of 64 steps::

                patch_sample = Patchify(64)(sample)

        """
        self._validate_sample(sample)
        types = sample.types
        coords = sample.coords
        seq_len = types.size(0)
        pad = (-seq_len) % self.patch_size
        num_patches = (seq_len + pad) // self.patch_size

        pad_types = torch.cat([types, types.new_zeros(pad)], 0)
        pad_coords = torch.cat([coords, coords.new_zeros(pad, coords.size(1))], 0)

        patch_types = pad_types.view(num_patches, self.patch_size)
        patch_coords = pad_coords.view(num_patches, self.patch_size, coords.size(1))

        sample_type = type(sample)
        return sample_type(
            types=patch_types,
            coords=patch_coords,
            metrics=sample.metrics,
            style_idx=sample.style_idx,
            content_idx=sample.content_idx,
        )


__all__ = [
    "Compose",
    "LimitSequenceLength",
    "Patchify",
    "QuadToCubic",
]
