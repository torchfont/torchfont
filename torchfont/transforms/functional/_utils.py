"""Registration and dispatch utilities for functional kernels."""

from __future__ import annotations

import functools
from typing import TYPE_CHECKING, Any, Concatenate, ParamSpec, TypeVar, cast

import torch

from torchfont.structures import COORD_DIM, Outline

if TYPE_CHECKING:
    from collections.abc import Callable

    import numpy as np
    from numpy.typing import NDArray

P = ParamSpec("P")
R = TypeVar("R")
S = TypeVar("S")

_KERNEL_REGISTRY: dict[object, dict[type[object], object]] = {}


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


def _passthrough(inpt: object, *_args: object, **_kwargs: object) -> object:
    return inpt


def _get_kernel(
    functional: object,
    input_type: type[object],
    *,
    allow_passthrough: bool = False,
) -> Callable[..., object]:
    registry = _KERNEL_REGISTRY.get(functional)
    if not registry:
        name = getattr(functional, "__name__", type(functional).__name__)
        msg = f"no kernel is registered for {name}"
        raise ValueError(msg)
    for cls in input_type.__mro__:
        if kernel := registry.get(cls):
            return cast("Callable[..., object]", kernel)
    if allow_passthrough:
        return _passthrough
    name = getattr(functional, "__name__", type(functional).__name__)
    msg = f"{name} does not support {input_type.__name__}"
    raise TypeError(msg)


def _dispatchable(
    input_type: type[S],
) -> Callable[[Callable[Concatenate[S, P], R]], Callable[Concatenate[S, P], R]]:
    def decorator(
        kernel: Callable[Concatenate[S, P], R],
    ) -> Callable[Concatenate[S, P], R]:
        @functools.wraps(kernel)
        def dispatcher(inpt: S, *args: P.args, **kwargs: P.kwargs) -> R:
            resolved = cast(
                "Callable[Concatenate[S, P], R]",
                _get_kernel(dispatcher, type(inpt)),
            )
            return resolved(inpt, *args, **kwargs)

        _KERNEL_REGISTRY[dispatcher] = {input_type: kernel}
        return dispatcher

    return decorator
