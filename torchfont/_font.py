"""Persistent font references."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from os import PathLike


@dataclass(frozen=True)
class FontRef:
    """Persistent file-local reference to one font."""

    path: str
    ttc_index: int

    def __init__(self, path: str | PathLike[str], ttc_index: int) -> None:
        object.__setattr__(self, "path", os.fspath(Path(path)))
        object.__setattr__(self, "ttc_index", ttc_index)


__all__ = ["FontRef"]
