"""Shared helpers for functional kernels."""

from __future__ import annotations

from typing import TYPE_CHECKING

from torchfont._outline import Outline

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor


def _require_no_grad(inpt: Outline, name: str) -> None:
    """Reject an outline that requires grad for a non-differentiable kernel.

    Kernels implemented in Rust change outline topology, so no gradient is
    defined for them. Failing here names the kernel instead of surfacing an
    error from deep inside the operator.
    """
    if inpt.coords.requires_grad or inpt.types.requires_grad:
        msg = (
            f"{name} is implemented in Rust and is not differentiable; "
            "detach its coords first"
        )
        raise RuntimeError(msg)


def _native_outline(
    inpt: Outline,
    operation: Callable[..., tuple[Tensor, Tensor]],
    *args: object,
    name: str,
) -> Outline:
    """Run a Rust outline operator, checking preconditions once at the boundary.

    ``operation`` is a :mod:`torchfont._ops` custom operator, so the Rust call is
    one opaque node that :func:`torch.compile` can capture.
    """
    _require_no_grad(inpt, name)
    out_types, out_coords = operation(inpt.types, inpt.coords, *args)
    return Outline._wrap(out_types, out_coords)  # noqa: SLF001


def _same_types(inpt: Outline, coords: Tensor) -> Outline:
    """Pair new coordinates with the element types of an existing outline."""
    return Outline._wrap(inpt.types, coords)  # noqa: SLF001
