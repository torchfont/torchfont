"""Structured metadata objects for glyph datasets."""

from typing import NamedTuple


class StyleLabel(NamedTuple):
    """Metadata for a style label stored on a dataset."""

    idx: int
    label_id: str
    name: str


class ContentLabel(NamedTuple):
    """Metadata for a content label stored on a dataset."""

    idx: int
    label_id: str
    char: str
    codepoint: int


class DatasetMetadata(NamedTuple):
    """Immutable-style label metadata for a glyph dataset.

    Attributes:
        styles (tuple[StyleLabel, ...]): Style label entries ordered by index.
        contents (tuple[ContentLabel, ...]): Content label entries ordered by
            index.
        style_id_to_idx (dict[str, int]): Mapping from style ``label_id`` to
            style index.
        style_name_to_idxs (dict[str, tuple[int, ...]]): Mapping from display
            name to all matching style indices.
        content_id_to_idx (dict[str, int]): Mapping from content ``label_id``
            to content index.

    """

    styles: tuple[StyleLabel, ...]
    contents: tuple[ContentLabel, ...]
    style_id_to_idx: dict[str, int]
    style_name_to_idxs: dict[str, tuple[int, ...]]
    content_id_to_idx: dict[str, int]


def build_dataset_metadata(
    style_names: list[str],
    content_codepoints: list[int],
) -> DatasetMetadata:
    """Build a metadata object from style names and Unicode codepoints."""
    styles = tuple(
        StyleLabel(idx=idx, label_id=f"style:{idx}", name=name)
        for idx, name in enumerate(style_names)
    )
    contents = tuple(
        ContentLabel(
            idx=idx,
            label_id=f"content:U+{cp:04X}",
            char=chr(cp),
            codepoint=cp,
        )
        for idx, cp in enumerate(content_codepoints)
    )

    grouped_names: dict[str, list[int]] = {}
    for label in styles:
        grouped_names.setdefault(label.name, []).append(label.idx)

    return DatasetMetadata(
        styles=styles,
        contents=contents,
        style_id_to_idx={label.label_id: label.idx for label in styles},
        style_name_to_idxs={name: tuple(idxs) for name, idxs in grouped_names.items()},
        content_id_to_idx={label.label_id: label.idx for label in contents},
    )
