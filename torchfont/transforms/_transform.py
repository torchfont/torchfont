"""Torchvision v2-style transform primitives for semantic font data."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar, Generic, TypeVar, cast

import torch
from torch import Tensor, nn
from torch._C import DisableTorchFunctionSubclass
from torch.utils._pytree import register_pytree_node, tree_flatten, tree_unflatten

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torchfont.datasets import GlyphSample, VariableGlyphSample

T = TypeVar("T")


@dataclass(frozen=True)
class Outline:
    """A glyph outline represented by coupled element-type and coordinate tensors."""

    types: Tensor
    coords: Tensor


_PRESERVE_TF_TENSOR_OPS = {
    Tensor.clone,
    Tensor.detach,
    Tensor.pin_memory,
    Tensor.requires_grad_,
    Tensor.to,
}


class TFTensor(Tensor):
    """Base class for tensors carrying TorchFont transform semantics.

    Ordinary tensor operations return plain tensors. Operations that only copy
    or relocate a tensor preserve its semantic subclass.
    """

    @classmethod
    def __torch_function__(
        cls,
        func: Callable[..., object],
        types: tuple[type[Tensor], ...],
        args: tuple[Any, ...] = (),
        kwargs: dict[str, Any] | None = None,
    ) -> object:
        if not all(issubclass(cls, tensor_type) for tensor_type in types):
            return NotImplemented
        with DisableTorchFunctionSubclass():
            output = func(*args, **(kwargs or {}))
        if func in _PRESERVE_TF_TENSOR_OPS and isinstance(args[0], cls):
            return cls._wrap_output(output)
        if isinstance(output, cls):
            return output.as_subclass(Tensor)
        return output

    @classmethod
    def _wrap_output(cls, output: object) -> object:
        if isinstance(output, Tensor) and not isinstance(output, cls):
            return output.as_subclass(cls)
        if isinstance(output, (tuple, list)):
            return type(output)(cls._wrap_output(item) for item in output)
        return output

    def __deepcopy__(self, memo: dict[int, Any]) -> TFTensor:
        del memo
        return cast(
            "TFTensor", self.detach().clone().requires_grad_(self.requires_grad)
        )


class Bitmap(TFTensor):
    """A rasterized glyph image distinguished from an ordinary tensor."""

    def __new__(cls: type[Bitmap], data: Tensor) -> Bitmap:  # noqa: PYI034
        return data.as_subclass(cls)


@dataclass(frozen=True)
class GlyphData(Generic[T]):
    """A transformed glyph payload together with its dataset metadata."""

    data: T
    sample: GlyphSample | VariableGlyphSample
    location: Mapping[str, float]


def _flatten_glyph_data(value: GlyphData[Any]) -> tuple[list[Any], object]:
    return [value.data], (value.sample, value.location)


def _unflatten_glyph_data(
    children: Iterable[Any], context: tuple[Any, Any]
) -> GlyphData[Any]:
    sample, location = context
    (data,) = children
    return GlyphData(data, sample, location)


register_pytree_node(GlyphData, _flatten_glyph_data, _unflatten_glyph_data)


class Transform(nn.Module):
    """Base class for type-directed transforms over nested pytree inputs."""

    _transformed_types: ClassVar[tuple[type[Any] | Callable[[object], bool], ...]] = (
        Outline,
    )

    def check_inputs(self, _flat_inputs: list[object]) -> None:
        """Check relationships between all inputs before sampling parameters."""

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        """Create parameters shared by all selected inputs in one call."""
        return {}

    def transform(self, inpt: object, params: dict[str, Any]) -> object:
        """Transform one selected input using parameters from ``make_params``."""
        raise NotImplementedError

    def forward(self, *inputs: object) -> object:
        """Transform selected leaves and preserve the enclosing pytree structure."""
        if not inputs:
            msg = "Transform requires at least one input"
            raise TypeError(msg)
        inpt = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, tree_spec = tree_flatten(inpt)
        self.check_inputs(flat_inputs)
        needs_transform = self._needs_transform_list(flat_inputs)
        selected = [
            item
            for item, selected in zip(flat_inputs, needs_transform, strict=True)
            if selected
        ]
        params = self.make_params(selected)
        flat_outputs = [
            self.transform(item, params) if selected else item
            for item, selected in zip(flat_inputs, needs_transform, strict=True)
        ]
        return tree_unflatten(flat_outputs, tree_spec)

    def _needs_transform_list(self, flat_inputs: list[Any]) -> list[bool]:
        """Select semantic leaves while passing all other inputs through."""
        return [
            any(
                spec(inpt) if not isinstance(spec, type) else isinstance(inpt, spec)
                for spec in self._transformed_types
            )
            for inpt in flat_inputs
        ]

    def extra_repr(self) -> str:
        """Show simple public configuration values in module representations."""
        printable = (bool, int, float, str, tuple, list, Enum)
        return ", ".join(
            f"{name}={value!r}"
            for name, value in self.__dict__.items()
            if not name.startswith("_")
            and name != "training"
            and isinstance(value, printable)
        )


class Compose(Transform):
    """Apply a sequence of transforms in order."""

    def __init__(self, transforms: Sequence[Callable[..., object]]) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            msg = "transforms must be a sequence of callables"
            raise TypeError(msg)
        if not transforms:
            msg = "transforms must not be empty"
            raise ValueError(msg)
        if not all(callable(transform) for transform in transforms):
            msg = "transforms must contain only callables"
            raise TypeError(msg)
        self.transforms = transforms

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms to ``inpt``."""
        unpack = len(inputs) > 1
        for transform in self.transforms:
            output = transform(*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output

    def extra_repr(self) -> str:
        return "\n".join(f"    {transform}" for transform in self.transforms)


class RandomApply(Transform):
    """Apply a sequence of transforms with probability ``p``."""

    def __init__(
        self, transforms: Sequence[Callable[..., object]], p: float = 0.5
    ) -> None:
        super().__init__()
        if not isinstance(transforms, Sequence):
            msg = "transforms must be a sequence of callables"
            raise TypeError(msg)
        if not transforms:
            msg = "transforms must not be empty"
            raise ValueError(msg)
        if not all(callable(transform) for transform in transforms):
            msg = "transforms must contain only callables"
            raise TypeError(msg)
        if not 0.0 <= p <= 1.0:
            msg = "p must be between 0 and 1"
            raise ValueError(msg)
        self.transforms = transforms
        self.p = p

    def forward(self, *inputs: object) -> object:
        """Apply all configured transforms together or return ``inpt`` unchanged."""
        if not inputs:
            msg = "RandomApply requires at least one input"
            raise TypeError(msg)
        unpack = len(inputs) > 1
        if torch.rand(()) >= self.p:
            return inputs if unpack else inputs[0]
        for transform in self.transforms:
            output = transform(*inputs)
            inputs = cast("tuple[object, ...]", output) if unpack else (output,)
        return output

    def extra_repr(self) -> str:
        transforms = "\n".join(f"    {transform}" for transform in self.transforms)
        return f"p={self.p}\n{transforms}"
