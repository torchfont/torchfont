"""Google Fonts Glyphsets.

Codepoint groups from the Google Fonts "glyphsets" repository.
Use these groups to filter a dataset by script or character coverage.
"""

from torchfont._torchfont import (
    LATIN_CORE,
    LATIN_KERNEL,
    get_glyphset_codepoints,
)

__all__ = [
    "LATIN_CORE",
    "LATIN_KERNEL",
    "get_glyphset_codepoints",
]
