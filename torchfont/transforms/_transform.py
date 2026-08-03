"""torchvision.transforms.v2-style primitives for semantic font data."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
from enum import Enum
from typing import TYPE_CHECKING, Any, ClassVar

from torch import nn
from torch.utils._pytree import tree_flatten, tree_unflatten

from torchfont._outline import Outline

if TYPE_CHECKING:
    from collections.abc import Iterator


_SHARED_PARAMS_TARGET: ContextVar[object | None] = ContextVar(
    "torchfont_shared_params_target", default=None
)


@contextmanager
def _share_params(transform: object) -> Iterator[None]:
    token = _SHARED_PARAMS_TARGET.set(transform)
    try:
        yield
    finally:
        _SHARED_PARAMS_TARGET.reset(token)


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
        return self._forward(inputs, same_params=_SHARED_PARAMS_TARGET.get() is self)

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
