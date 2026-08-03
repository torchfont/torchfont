"""Shared helpers for functional kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import torch

from torchfont.structures import COORD_DIM, Outline

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray


def _native_outline(
    inpt: Outline,
    operation: Callable[
        ..., tuple[NDArray[np.integer[Any]], NDArray[np.floating[Any]]]
    ],
    *args: object,
) -> Outline:
    """Run a native outline operation on CPU and restore the input devices."""
    types_device = inpt.types.device
    coords_device = inpt.coords.device
    types = inpt.types.cpu().contiguous()
    coords = inpt.coords.cpu().contiguous()
    out_types, out_coords = operation(types.numpy(), coords.reshape(-1).numpy(), *args)
    return Outline(
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, COORD_DIM).to(device=coords_device),
    )
