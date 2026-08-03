"""torchvision.transforms.v2-style primitives for semantic font data."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, ClassVar

from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchfont.structures import Outline


@dataclass(frozen=True)
class _SharedParamsInput:
    inputs: tuple[object, ...]


class Transform(nn.Module):
    """Base class for type-directed transforms over nested pytree inputs."""

    _transformed_types: ClassVar[tuple[type[Any], ...]] = (Outline,)

    def check_inputs(self, _flat_inputs: list[object]) -> None:
        """Check relationships between all inputs before sampling parameters."""

    def check_same_params(self, _selected_inputs: list[object]) -> None:
        """Check inputs before explicitly sharing one parameter sample."""

    def make_params(self, _flat_inputs: list[Any]) -> dict[str, Any]:
        """Create parameters for selected inputs in one sampling group."""
        return {}

    def transform(self, inpt: object, params: dict[str, Any]) -> object:
        """Transform one selected input using parameters from ``make_params``."""
        raise NotImplementedError

    def forward(self, *inputs: object) -> object:
        """Transform semantic leaves and preserve the enclosing pytree."""
        if len(inputs) == 1 and isinstance(inputs[0], _SharedParamsInput):
            return self._forward(inputs[0].inputs, same_params=True)
        return self._forward(inputs, same_params=False)

    def _forward(self, inputs: tuple[object, ...], *, same_params: bool) -> object:
        inpt = inputs if len(inputs) > 1 else inputs[0]
        flat_inputs, tree_spec = tree_flatten(inpt)
        self.check_inputs(flat_inputs)
        needs_transform = self._needs_transform_list(flat_inputs)
        selected = [
            item
            for item, selected in zip(flat_inputs, needs_transform, strict=True)
            if selected
        ]
        if same_params:
            self.check_same_params(selected)
        shared_params = self.make_params(selected) if same_params and selected else None
        flat_outputs = []
        for item, selected in zip(flat_inputs, needs_transform, strict=True):
            if not selected:
                flat_outputs.append(item)
                continue
            params = (
                shared_params if shared_params is not None else self.make_params([item])
            )
            flat_outputs.append(self.transform(item, params))
        return tree_unflatten(flat_outputs, tree_spec)

    def _needs_transform_list(self, flat_inputs: list[Any]) -> list[bool]:
        """Select semantic leaves while passing all other inputs through."""
        return [isinstance(inpt, self._transformed_types) for inpt in flat_inputs]

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
