"""Subpath start-point transformation functions."""

import torch
from torch import Tensor

from torchfont import _torchfont


def normalize_subpath_start_points(
    types: Tensor,
    coords: Tensor,
) -> tuple[Tensor, Tensor]:
    """Move each subpath start to its lexicographically smallest endpoint.

    ``(x, y)`` endpoint order is used as the deterministic key. Open subpaths
    (those without a closing ``Close``), ``END``, and ``PAD`` element types are
    returned unchanged. When rotation crosses the old closing edge, that implicit
    edge is materialised as ``LINE_TO`` so the represented geometry is preserved.
    """
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    out_types, out_coords = _torchfont.normalize_subpath_start_points(
        types.numpy(), coords.reshape(-1).numpy()
    )
    return (
        torch.from_numpy(out_types),
        torch.from_numpy(out_coords).view(-1, 6),
    )


def randomize_subpath_start_points(
    types: Tensor,
    coords: Tensor,
    *,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    """Choose a uniformly random start endpoint for each subpath.

    Open subpaths (those without a closing ``Close``), ``END``, and ``PAD``
    element types are returned unchanged. Pass a ``torch.Generator`` to make the
    independent per-subpath choices reproducible.
    """
    types = types.cpu().contiguous()
    coords = coords.cpu().contiguous()
    random_values = torch.rand(
        types.size(0),
        device=generator.device if generator is not None else types.device,
        generator=generator,
    ).cpu()
    out_types, out_coords = _torchfont.randomize_subpath_start_points(
        types.numpy(),
        coords.reshape(-1).numpy(),
        random_values.numpy(),
    )
    return (
        torch.from_numpy(out_types),
        torch.from_numpy(out_coords).view(-1, 6),
    )
