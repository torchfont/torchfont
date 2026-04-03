from collections.abc import Sequence

class FontDataset:
    def __init__(
        self,
        root: str,
        codepoints: Sequence[int] | None = ...,
        patterns: Sequence[str] | None = ...,
    ) -> None: ...

    sample_count: int
    style_class_count: int
    content_class_count: int

    content_classes: list[int]

    style_classes: list[str]
    style_rows: list[tuple[str, str, int, int | None]]

    def item(
        self,
        idx: int,
    ) -> tuple[list[int], list[float], int, int]: ...
    def locate(
        self,
        idx: int,
    ) -> tuple[str, int, int | None, int, int, int]: ...
    def targets(self) -> bytes: ...
