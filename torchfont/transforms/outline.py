"""Whole-outline transformation functions."""

import torch
from torch import Tensor

from torchfont import _torchfont


def remove_overlaps(types: Tensor, coords: Tensor) -> tuple[Tensor, Tensor]:
    """Merge overlapping subpaths using Skia PathOps winding simplification.

    Args:
        types: 1-D ``torch.int64`` tensor of element types.
        coords: 2-D ``torch.float32`` tensor of shape ``(N, 6)``.

    Returns:
        A new variable-length outline tuple ``(types, coords)`` with overlapping
        subpath edges removed when Skia PathOps can resolve the outline. If
        PathOps cannot simplify an otherwise valid outline, the original outline
        is returned unchanged.

    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.remove_overlaps(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )


def random_remove_overlaps(
    types: Tensor,
    coords: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Randomly merge one or more groups of potentially overlapping subpaths.

    Tight bounding-box intersections are used as an inexpensive approximation of
    overlap. Bbox-connected components form independent groups, each selected with
    probability 0.5 and simplified with one Skia PathOps call. If overlap groups
    exist, at least one is selected. If PathOps cannot simplify a selected group,
    that group is returned unchanged. Pass a ``torch.Generator`` to make the
    selection reproducible.
    """
    types_device = types.device
    coords_device = coords.device
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    random_values = torch.rand(
        types.size(0),
        device=generator.device if generator is not None else types.device,
        generator=generator,
    ).cpu()
    out_types, out_coords = _torchfont.random_remove_overlaps(
        types.numpy(), coords.reshape(-1).numpy(), random_values.numpy()
    )
    return (
        torch.from_numpy(out_types).to(device=types_device),
        torch.from_numpy(out_coords).view(-1, 6).to(device=coords_device),
    )
