"""Structured metadata objects for glyph datasets."""

from typing import NamedTuple


class StyleAxis(NamedTuple):
    """One user-space variation axis value for a style."""

    tag: str
    value: float


class StyleLabel(NamedTuple):
    """Metadata for a style label stored on a dataset."""

    idx: int
    label_id: str
    name: str
    axes: tuple[StyleAxis, ...]


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


StyleMetadataRow = tuple[str, str, tuple[StyleAxis, ...]]
ContentMetadataRow = tuple[str, str, int]


def build_dataset_metadata(
    style_rows: list[StyleMetadataRow],
    content_rows: list[ContentMetadataRow],
) -> DatasetMetadata:
    """Build a DatasetMetadata object for a glyph dataset.

    This constructs style and content label entries and their lookup tables.
    Style rows already carry precomputed, collision-safe ``label_id`` values.

    Args:
        style_rows: Tuples of ``(name, label_id, axes)`` aligned to the
            dataset's style indices.
        content_rows: Tuples of ``(label_id, char, codepoint)`` aligned to the
            dataset's content indices.

    Returns:
        DatasetMetadata: Immutable metadata containing style and content label
        entries and their associated lookup dictionaries.

    """
    styles = tuple(
        StyleLabel(
            idx=idx,
            label_id=label_id,
            name=name,
            axes=axes,
        )
        for idx, (name, label_id, axes) in enumerate(style_rows)
    )
    contents = tuple(
        ContentLabel(
            idx=idx,
            label_id=label_id,
            char=char,
            codepoint=codepoint,
        )
        for idx, (label_id, char, codepoint) in enumerate(content_rows)
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
