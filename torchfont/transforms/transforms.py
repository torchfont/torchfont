"""Composable transforms for polishing tensor glyph sequences before training.

Notes:
    Each transform is intentionally small and deterministic, allowing you to mix
    and match them in :class:`torch.utils.data.Dataset` pipelines without losing
    track of where normalization occurs.

Examples:
    Normalize and patch glyph tensors in a single pipeline::

        pipeline = Compose([QuadToCubic(), LimitSequenceLength(256), Patchify(32)])

"""

from collections.abc import Callable, Sequence

import torch
from torch import Tensor

from torchfont.io.outline import CommandType


class Compose:
    """Apply a curated list of transform callables to every sample.

    See Also:
        LimitSequenceLength: Useful when a hard cap is required before adding
        more expensive stages.

    """

    def __init__(
        self,
        transforms: Sequence[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]],
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

    def __call__(self, types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Apply every transform in order to the provided sample.

        Args:
            types (Tensor): Input command sequence.
            coords (Tensor): Input coordinate sequence.

        Returns:
            tuple[Tensor, Tensor]: Resulting sample after all transformations
            are applied.

        Examples:
            Run the composed pipeline on a glyph sample::

                types, coords = pipeline(types, coords)

        """
        for t in self.transforms:
            types, coords = t(types, coords)
        return types, coords


class LimitSequenceLength:
    """Trim glyph sequences to a predictable maximum length.

    See Also:
        Patchify: Converts sequences into fixed-size blocks after truncation so
        transformer-style models see a uniform layout.

    """

    def __init__(self, max_len: int) -> None:
        """Initialize the transform with the desired maximum length.

        Args:
            max_len (int): Maximum number of time steps to keep. Any surplus
                command or coordinate pairs are discarded.

        Examples:
            Cap sequences at 512 steps::

                LimitSequenceLength(512)

        """
        self.max_len = max_len

    def __call__(self, types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Clip the sequence and coordinate tensors to the specified length.

        Args:
            types (Tensor): Tensor of pen command types.
            coords (Tensor): Tensor of pen command coordinates.

        Returns:
            tuple[Tensor, Tensor]: Tensors truncated to ``max_len`` elements.

        Warnings:
            Elements beyond ``max_len`` are removed rather than padded or
            aggregated, so downstream code should account for the shorter tail.

        Examples:
            Clamp a sample to 128 steps::

                types, coords = LimitSequenceLength(128)(types, coords)

        """
        types = types[: self.max_len]
        coords = coords[: self.max_len]

        return types, coords


class QuadToCubic:
    """Convert quadratic segments into cubic segments in tensor form.

    Notes:
        This transform rewrites only `CommandType.QUAD_TO` rows and leaves all
        other commands untouched. Coordinate dimensionality stays at 6.

    """

    def __call__(self, types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Convert `QUAD_TO` entries to `CURVE_TO`.

        Args:
            types (Tensor): Command tensor whose values follow `CommandType`.
            coords (Tensor): Coordinate tensor with the last dimension at least
                6 and layout `[cx0, cy0, cx1, cy1, x, y]`.

        Returns:
            tuple[Tensor, Tensor]: Converted `(types, coords)` tensors.

        """
        flat_types = types.reshape(-1)
        flat_coords = coords.reshape(-1, coords.size(-1))
        quad = flat_types == CommandType.QUAD_TO.value

        if not torch.any(quad):
            return types, coords

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

        return out_types.view_as(types), out_coords.view_as(coords)


class Patchify:
    """Pad glyph sequences and reshape them into uniform, model-friendly patches.

    See Also:
        LimitSequenceLength: Apply beforehand when you need a strict ceiling on
        the number of emitted patches.

    """

    def __init__(self, patch_size: int) -> None:
        """Configure the patch length for reshaping sequences.

        Args:
            patch_size (int): Number of time steps captured in each patch. Choose
                values that align with the receptive field of your downstream
                model.

        Examples:
            Create 32-step patches for transformer models::

                Patchify(32)

        """
        self.patch_size = patch_size

    def __call__(self, types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
        """Pad and reshape sequences into contiguous patches.

        Args:
            types (Tensor): Tensor of pen command types.
            coords (Tensor): Tensor of pen command coordinates.

        Returns:
            tuple[Tensor, Tensor]: Tensors grouped into patches of
            ``patch_size`` steps. Trailing zeros are added only when needed for
            alignment.

        Tips:
            Pair with :class:`LimitSequenceLength` to bound the worst-case number
            of patches in a batch.

        Examples:
            Reshape a glyph sequence into patches of 64 steps::

                patch_types, patch_coords = Patchify(64)(types, coords)

        """
        seq_len = types.size(0)
        pad = (-seq_len) % self.patch_size
        num_patches = (seq_len + pad) // self.patch_size

        pad_types = torch.cat([types, types.new_zeros(pad)], 0)
        pad_coords = torch.cat([coords, coords.new_zeros(pad, coords.size(1))], 0)

        patch_types = pad_types.view(num_patches, self.patch_size)
        patch_coords = pad_coords.view(num_patches, self.patch_size, coords.size(1))

        return patch_types, patch_coords
