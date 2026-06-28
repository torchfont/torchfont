"""Variation instantiation policies for variable fonts."""

from typing import TypeAlias

from torchfont._torchfont import (
    DefaultInstantiation,
    GridInstantiation,
    NamedInstantiation,
)

VariationInstantiation: TypeAlias = (
    DefaultInstantiation | NamedInstantiation | GridInstantiation
)

__all__ = [
    "DefaultInstantiation",
    "GridInstantiation",
    "NamedInstantiation",
    "VariationInstantiation",
]
