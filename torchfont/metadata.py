"""Structured metadata objects for glyph datasets."""

from pathlib import Path
from typing import NamedTuple
from urllib.parse import quote


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


StyleMetadataRow = tuple[str, str | Path, int, int | None]


def build_dataset_metadata(
    root: Path,
    style_rows: list[StyleMetadataRow],
    content_codepoints: list[int],
) -> DatasetMetadata:
    """Build a DatasetMetadata object for a glyph dataset.

    This constructs style and content label entries and their lookup tables.
    Style ``label_id`` values are derived from the underlying font file
    locations (relative to ``root``) and the ``(face_idx, instance_idx)``
    values in ``style_rows``, via :func:`_style_label_id`.

    Args:
        root: Common root directory used to relativize font paths when
            constructing stable style ``label_id`` values.  Every font path
            in ``style_rows`` must be located inside ``root``; a
            ``ValueError`` is raised if any path falls outside it.
        style_rows: Tuples of ``(name, font_path, face_idx, instance_idx)``
            aligned to the dataset's style indices. ``instance_idx`` is
            ``None`` for static faces.
        content_codepoints: Unicode code points to turn into ``ContentLabel``
            entries.

    Returns:
        DatasetMetadata: Immutable metadata containing style and content label
        entries and their associated lookup dictionaries.

    Raises:
        ValueError: If any font path in ``style_rows`` is not located under
            ``root``.

    """
    styles = tuple(
        StyleLabel(
            idx=idx,
            label_id=_style_label_id(
                root,
                Path(font_path),
                face_idx,
                instance_idx,
            ),
            name=name,
        )
        for idx, (name, font_path, face_idx, instance_idx) in enumerate(style_rows)
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


def _style_label_id(
    root: Path,
    font_path: Path,
    face_idx: int,
    instance_idx: int | None,
) -> str:
    """Build a stable, source-based style label ID.

    Raises:
        ValueError: If ``font_path`` is not located under ``root``.

    """
    try:
        relative_path = font_path.relative_to(root)
    except ValueError:
        msg = f"font path {font_path!r} is not under dataset root {root!r}"
        raise ValueError(msg) from None

    instance_value = "static" if instance_idx is None else str(instance_idx)
    quoted_path = quote(relative_path.as_posix(), safe="/")
    return f"style:path={quoted_path};face={face_idx};instance={instance_value}"
