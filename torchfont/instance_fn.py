"""Instance-location and instance-count functions for glyph datasets."""

from collections.abc import Callable, Mapping, Sequence
from operator import index
from typing import TYPE_CHECKING, TypeAlias

from torchfont import _torchfont

if TYPE_CHECKING:
    from torchfont.datasets import FontRef

InstanceLocationsFn: TypeAlias = Callable[["FontRef"], Sequence[Mapping[str, float]]]
InstanceCountFn: TypeAlias = Callable[["FontRef"], int]


def default_instance(font: "FontRef") -> list[dict[str, float]]:
    """Return the font's default variation location."""
    return [
        _location_dict(_torchfont.default_location_for_font(font.path, font.ttc_index))
    ]


def named_instances(font: "FontRef") -> list[dict[str, float]]:
    """Return deduplicated named instances, or the default location if absent."""
    locations = _torchfont.named_instance_locations_for_font(font.path, font.ttc_index)
    if not locations:
        return default_instance(font)
    return [_location_dict(location) for location in locations]


def grid_instances(axes: Mapping[str, int]) -> InstanceLocationsFn:
    """Build an instance function that samples selected axes on an even grid."""
    counts = {str(tag): index(count) for tag, count in axes.items()}

    def instances(font: "FontRef") -> list[dict[str, float]]:
        return [
            _location_dict(location)
            for location in _torchfont.grid_locations_for_font(
                font.path,
                font.ttc_index,
                counts,
            )
        ]

    return instances


def default_instance_count(_font: "FontRef") -> int:
    """Return one instance slot for a font."""
    return 1


def named_instance_count(font: "FontRef") -> int:
    """Return the named-instance count, or one if absent."""
    return len(named_instances(font))


def grid_instance_count(axes: Mapping[str, int]) -> InstanceCountFn:
    """Build an instance-count function matching ``grid_instances``."""
    counts = {str(tag): index(count) for tag, count in axes.items()}

    def count(font: "FontRef") -> int:
        return int(
            _torchfont.grid_location_count_for_font(
                font.path,
                font.ttc_index,
                counts,
            ),
        )

    return count


def _location_dict(pairs: Sequence[tuple[str, float]]) -> dict[str, float]:
    return {str(tag): float(value) for tag, value in pairs}


__all__ = [
    "InstanceCountFn",
    "InstanceLocationsFn",
    "default_instance",
    "default_instance_count",
    "grid_instance_count",
    "grid_instances",
    "named_instance_count",
    "named_instances",
]
