"""Internal dataset normalization helpers."""

from __future__ import annotations

from operator import index
from typing import TYPE_CHECKING, SupportsIndex

import numpy as np
import torch

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor


def normalize_patterns(
    patterns: str | Sequence[str] | None,
) -> tuple[str, ...] | None:
    if patterns is None:
        return None
    if isinstance(patterns, str):
        return (patterns,)
    return tuple(str(pattern) for pattern in patterns)


def normalize_codepoints(
    codepoints: Sequence[SupportsIndex] | None,
) -> tuple[int, ...] | None:
    if codepoints is None:
        return None
    return tuple(sorted({index(codepoint) for codepoint in codepoints}))


def normalize_max_length(max_length: SupportsIndex | None) -> int | None:
    if max_length is None:
        return None
    return index(max_length)


def normalize_index(idx: SupportsIndex, dataset_len: int) -> int:
    resolved_idx = index(idx)
    original_idx = resolved_idx
    if resolved_idx < 0:
        resolved_idx += dataset_len
    if resolved_idx < 0 or resolved_idx >= dataset_len:
        msg = (
            f"index {original_idx} is out of range for dataset of length {dataset_len}"
        )
        raise IndexError(msg)
    return resolved_idx


def font_targets_from_offsets(offsets: tuple[int, ...]) -> Tensor:
    """Repeat each face index across the sample range that face owns."""
    bounds = np.asarray(offsets, dtype=np.int64)
    faces = np.arange(len(offsets) - 1, dtype=np.int64)
    return torch.from_numpy(np.repeat(faces, np.diff(bounds)))
