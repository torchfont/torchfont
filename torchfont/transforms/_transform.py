"""Torchvision v2-style transform primitives for semantic font data."""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchfont.structures import Outline
from torchfont.transforms.functional._utils import _get_kernel

if TYPE_CHECKING:
    from collections.abc import Callable


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

    def _call_kernel(
        self,
        functional: Callable[..., object],
        inpt: object,
        *args: object,
        **kwargs: object,
    ) -> object:
        """Dispatch a functional kernel or pass unsupported semantic data through."""
        kernel = _get_kernel(functional, type(inpt), allow_passthrough=True)
        return kernel(inpt, *args, **kwargs)

    def forward(self, *inputs: object) -> object:
        """Transform selected leaves and preserve the enclosing pytree structure."""
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
            f"{name}={value}"
            for name, value in self.__dict__.items()
            if not name.startswith("_")
            and name != "training"
            and isinstance(value, printable)
        )
