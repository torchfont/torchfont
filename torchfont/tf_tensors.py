"""Semantic tensor subclasses used by TorchFont transforms."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, TypeVar, cast

import torch
from torch import Tensor
from torch._C import DisableTorchFunctionSubclass

_T = TypeVar("_T", bound="TFTensor")

if TYPE_CHECKING:
    from collections.abc import Callable

    from typing_extensions import Self

_PRESERVE_SUBCLASS_OPS = {
    Tensor.clone,
    Tensor.detach,
    Tensor.pin_memory,
    Tensor.requires_grad_,
    Tensor.to,
}


class TFTensor(Tensor):
    """Base class for tensors carrying TorchFont transform semantics.

    Native tensor operations return plain tensors by default. ``clone()``,
    ``detach()``, ``pin_memory()``, ``requires_grad_()``, and ``to()`` preserve
    the semantic subclass, matching torchvision's ``TVTensor`` convention.
    """

    @staticmethod
    def _to_tensor(
        data: Any,  # noqa: ANN401
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Tensor:
        if requires_grad is None:
            requires_grad = data.requires_grad if isinstance(data, Tensor) else False
        return torch.as_tensor(data, dtype=dtype, device=device).requires_grad_(
            requires_grad
        )

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type[Tensor], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        """Return plain tensors for native operations except copy-like methods."""
        if not all(issubclass(cls, tensor_type) for tensor_type in types):
            return NotImplemented
        with DisableTorchFunctionSubclass():
            output = func(*args, **(kwargs or {}))
        if func in _PRESERVE_SUBCLASS_OPS and isinstance(args[0], cls):
            return cls._wrap_output(output, like=args[0])
        if isinstance(output, cls):
            return output.as_subclass(Tensor)
        return output

    @classmethod
    def _wrap_output(cls, output: object, *, like: TFTensor) -> object:
        if isinstance(output, Tensor) and not isinstance(output, cls):
            return cls.wrap(output, like=like)
        if isinstance(output, (tuple, list)):
            return type(output)(cls._wrap_output(item, like=like) for item in output)
        return output

    @classmethod
    def wrap(
        cls,
        tensor: Tensor,
        *,
        like: Self,
        **metadata: object,
    ) -> Self:
        """Restore this semantic subclass, customizable for subclass metadata."""
        del like, metadata
        return cast("Self", tensor.as_subclass(cls))

    def __deepcopy__(self: _T, memo: dict[int, Any]) -> _T:  # noqa: PYI019
        """Copy tensor storage while preserving the semantic subclass."""
        del memo
        return cast("_T", self.detach().clone().requires_grad_(self.requires_grad))

    @property
    def shape(self) -> torch.Size:  # type: ignore[override]
        """Return the shape without semantic dispatch."""
        with DisableTorchFunctionSubclass():
            return super().shape

    @property
    def ndim(self) -> int:  # type: ignore[override]
        """Return the dimension count without semantic dispatch."""
        with DisableTorchFunctionSubclass():
            return super().ndim

    @property
    def device(self) -> torch.device:  # type: ignore[override]
        """Return the device without semantic dispatch."""
        with DisableTorchFunctionSubclass():
            return super().device

    @property
    def dtype(self) -> torch.dtype:  # type: ignore[override]
        """Return the dtype without semantic dispatch."""
        with DisableTorchFunctionSubclass():
            return super().dtype


class Bitmap(TFTensor):
    """A rasterized glyph image distinguished from an ordinary tensor."""

    def __new__(  # noqa: PYI034
        cls: type[Bitmap],
        data: Any,  # noqa: ANN401
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> Bitmap:
        """Construct a bitmap from tensor-like data."""
        tensor = cls._to_tensor(
            data, dtype=dtype, device=device, requires_grad=requires_grad
        )
        return tensor.as_subclass(cls)

    def __init__(
        self,
        data: Any,  # noqa: ANN401
        *,
        dtype: torch.dtype | None = None,
        device: torch.device | str | int | None = None,
        requires_grad: bool | None = None,
    ) -> None:
        """Accept the constructor arguments handled by ``__new__``."""
        del data, dtype, device, requires_grad


def wrap(wrappee: Tensor, *, like: _T, **metadata: object) -> _T:
    """Wrap a tensor like ``like``, preserving subclass-specific metadata."""
    return type(like).wrap(wrappee, like=like, **metadata)


__all__ = ["Bitmap", "TFTensor", "wrap"]
