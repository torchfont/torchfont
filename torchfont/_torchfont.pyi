from collections.abc import Sequence

class GlyphItem:
    types: bytes
    coords: bytes
    style_idx: int
    content_idx: int
    metrics: bytes
    glyph_name: str

class GlyphDataset:
    def __init__(
        self,
        root: str,
        codepoints: Sequence[int] | None = ...,
        patterns: Sequence[str] | None = ...,
    ) -> None: ...

    sample_count: int
    style_class_count: int
    content_class_count: int

    def content_metadata_rows(self) -> list[tuple[str, str, int]]: ...
    def style_metadata_rows(self, root: str) -> list[tuple[str, str]]: ...
    style_axes: list[list[tuple[str, float]]]
    def item(self, idx: int) -> GlyphItem: ...
    def targets(self) -> bytes: ...
