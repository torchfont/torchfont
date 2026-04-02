"""Utilities for turning local font folders into indexed glyph datasets.

Notes:
    Glyph data is cached inside the native backend for the lifetime of each
    dataset instance. Recreate the dataset when editing font files on disk to
    ensure changes are observed.

Examples:
    Iterate glyph samples from a directory of fonts::

        from torchfont.datasets import GlyphDataset

        dataset = GlyphDataset(root="~/fonts")
        sample = dataset[0]
        print(sample.types, sample.style_idx)

"""

from collections.abc import Callable, Sequence
from operator import index
from pathlib import Path
from typing import NamedTuple, SupportsIndex

import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont.io import COORD_DIM
from torchfont.metadata import (
    ContentLabel,
    DatasetMetadata,
    StyleLabel,
    build_dataset_metadata,
)


class GlyphSample(NamedTuple):
    """One glyph sample returned by a dataset.

    Using a NamedTuple means fields can be accessed by name rather than by
    position, so downstream code does not depend on tuple ordering.
    PyTorch's default DataLoader collation handles NamedTuples natively.

    Attributes:
        types (Tensor): 1-D long tensor of pen command types.
        coords (Tensor): 2-D float tensor of shape ``(N, 6)`` holding the
            coordinate data for each command.
        style_idx (int): Index into the dataset's ``style_classes`` list.
            When batches are formed by PyTorch's default DataLoader collation,
            this field becomes a 1-D ``torch.LongTensor``.
        content_idx (int): Index into the dataset's ``content_classes`` list.
            When batches are formed by PyTorch's default DataLoader collation,
            this field becomes a 1-D ``torch.LongTensor``.

    Examples:
        Access fields by name rather than by position::

            sample = dataset[0]
            print(sample.types.shape, sample.style_idx)

    """

    types: Tensor
    coords: Tensor
    style_idx: int
    content_idx: int


class GlyphLocation(NamedTuple):
    """Source metadata for one dataset index.

    Attributes:
        font_path (Path): Resolved path to the font file containing the glyph.
        face_idx (int): Zero-based face index within the font file. Collection
            formats such as TTC/OTC can expose multiple faces from one file.
        instance_idx (int | None): Zero-based named-instance index for variable
            fonts, or ``None`` for static fonts.
        codepoint (int): Unicode codepoint for the indexed glyph sample.
        style_idx (int): Index into the dataset's ``style_classes`` list.
        content_idx (int): Index into the dataset's ``content_classes`` list.

    Examples:
        Inspect where the first sample came from::

            location = dataset.locate(0)
            print(location.font_path, hex(location.codepoint))

    """

    font_path: Path
    face_idx: int
    instance_idx: int | None
    codepoint: int
    style_idx: int
    content_idx: int


class GlyphDataset(Dataset[GlyphSample]):
    """Dataset that yields glyph samples from a directory of font files.

    The dataset flattens every available code point and variation instance into
    a single indexable sequence. Each item returns the loader output along with
    style and content targets.

    Attributes:
        targets (torch.LongTensor): Label matrix of shape ``(N, 2)`` where
            column 0 holds the style class index and column 1 holds the
            content class index for every sample.
        content_classes (list[str]): List of Unicode character strings, one per
            content class, sorted by index. Use len(content_classes) to get
            the total number of content classes.
        content_class_to_idx (dict[str, int]): Mapping from characters to content
            class indices.
        style_classes (list[str]): List of style instance names, one per style
            class, sorted by index. Use len(style_classes) to get the total
            number of style classes.
        metadata (DatasetMetadata): Structured label metadata object that
            consolidates style/content labels and related lookup tables.
        style_labels (list[StyleLabel]): Collision-safe style metadata with
            explicit label IDs.
        style_label_to_idx (dict[str, int]): Mapping from style label IDs to
            style class indices.
        style_name_to_idxs (dict[str, list[int]]): Mapping from style names to
            all matching style indices.
        content_labels (list[ContentLabel]): Content metadata with stable label
            IDs and codepoints.
        content_label_to_idx (dict[str, int]): Mapping from content label IDs
            to content class indices.
        root (Path): Resolved root directory used for font discovery.
        patterns (tuple[str, ...] | None): Canonicalized path filter patterns.
        codepoints (tuple[int, ...] | None): Sorted unique Unicode code points
            applied during indexing.

    """

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: (Callable[[GlyphSample], GlyphSample] | None) = None,
    ) -> None:
        """Initialize the dataset by scanning font files and indexing samples.

        Args:
            root (Path | str): Directory containing font files. Both OTF and TTF
                files are discovered recursively.
            codepoints (Sequence[SupportsIndex] | None): Optional iterable
                of Unicode code points used to restrict the dataset content.
                Duplicate values are ignored and the effective filter is stored
                as sorted unique integers on ``dataset.codepoints``.
            patterns (Sequence[str] | None): Optional gitignore-style patterns
                describing which font paths to include.
            transform (Callable[[GlyphSample], GlyphSample] | None):
                Optional transformation applied to each sample before the item
                is returned.

        Examples:
            Restrict the dataset to uppercase ASCII glyphs::

                dataset = GlyphDataset(
                    root="~/fonts",
                    codepoints=[ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                )

        """
        self.root = Path(root).expanduser().resolve()
        self._validate_root_dir(self.root)
        self.transform = transform
        self.patterns = (
            tuple(str(pattern) for pattern in patterns)
            if patterns is not None
            else None
        )
        self.codepoints = self._normalize_codepoints(codepoints)

        self._dataset = _torchfont.FontDataset(
            str(self.root),
            self.codepoints,
            self.patterns,
        )
        self._metadata: DatasetMetadata | None = None

    def __repr__(self) -> str:
        """Return a human-readable summary of this dataset.

        Returns:
            str: String showing the class name, root path, sample count,
                number of style classes, and number of content classes.

        Examples:
            >>> ds = GlyphDataset(
            ...     root="tests/fonts",
            ...     codepoints=range(0x41, 0x5B),
            ...     patterns=("**/Lato-Regular.ttf",),
            ... )
            >>> len(ds), len(ds.style_classes), len(ds.content_classes)
            (26, 1, 26)

        """
        return (
            f"{type(self).__name__}("
            f"root={str(self.root)!r}, "
            f"samples={len(self)}, "
            f"styles={self._dataset.style_class_count}, "
            f"content_classes={self._dataset.content_class_count})"
        )

    def __getstate__(self) -> dict[str, object]:
        """Return state without the native backend for worker reconstruction."""
        state = self.__dict__.copy()
        state.pop("_dataset", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and recreate the native backend after unpickling."""
        self.__dict__.update(state)
        self._validate_root_dir(self.root)
        self._dataset = _torchfont.FontDataset(
            str(self.root),
            self.codepoints,
            self.patterns,
        )
        if not hasattr(self, "_metadata"):
            self._metadata = None

    @staticmethod
    def _validate_root_dir(root: Path) -> None:
        """Raise ValueError if *root* exists but is not a directory."""
        if root.exists() and not root.is_dir():
            msg = f"root must be a directory: {root}"
            raise ValueError(msg)

    @staticmethod
    def _normalize_codepoints(
        codepoints: Sequence[SupportsIndex] | None,
    ) -> tuple[int, ...] | None:
        """Convert an optional codepoint filter into a canonical tuple."""
        if codepoints is None:
            return None
        return tuple(sorted({index(cp) for cp in codepoints}))

    def _normalize_index(self, idx: SupportsIndex) -> int:
        """Resolve one dataset index, including negative indices."""
        resolved_idx = index(idx)
        original_idx = resolved_idx
        dataset_len = len(self)
        if resolved_idx < 0:
            resolved_idx += dataset_len
        if resolved_idx < 0 or resolved_idx >= dataset_len:
            msg = (
                f"index {original_idx} is out of range for dataset of length "
                f"{dataset_len}"
            )
            raise IndexError(msg)
        return resolved_idx

    def __len__(self) -> int:
        """Return the total number of glyph samples discoverable in the dataset.

        Returns:
            int: Total number of glyph samples available in the dataset.

        """
        return int(self._dataset.sample_count)

    def __getitem__(self, idx: SupportsIndex) -> GlyphSample:
        """Load a glyph sample and its associated targets.

        Args:
            idx (SupportsIndex): Zero-based index locating a sample across all
                fonts, code points, and instances. Any object implementing
                ``__index__`` is accepted. Negative indices are supported and
                count from the end of the dataset.

        Returns:
            GlyphSample: Structured sample containing ``types``, ``coords``,
            ``style_idx``, and ``content_idx``.

        Examples:
            Retrieve the first glyph sample and its target labels::

                sample = dataset[0]
                print(sample.types, sample.style_idx)

            Retrieve the last glyph sample::

                sample = dataset[-1]

        """
        idx = self._normalize_index(idx)
        raw_types, raw_coords, style_idx, content_idx = self._dataset.item(idx)
        types = torch.as_tensor(raw_types, dtype=torch.long)
        coords = torch.as_tensor(raw_coords, dtype=torch.float32).view(-1, COORD_DIM)
        sample = GlyphSample(
            types=types,
            coords=coords,
            style_idx=int(style_idx),
            content_idx=int(content_idx),
        )
        if self.transform is not None:
            return self.transform(sample)
        return sample

    def locate(self, idx: SupportsIndex) -> GlyphLocation:
        """Return source metadata for one dataset index.

        Args:
            idx (SupportsIndex): Zero-based index locating a sample across all
                fonts, faces, variation instances, and codepoints. Negative
                indices are supported and count from the end of the dataset.

        Returns:
            GlyphLocation: Source metadata describing the underlying font file,
            face, variation instance, Unicode codepoint, and label indices.

        Examples:
            Inspect the first sample's origin::

                location = dataset.locate(0)
                print(location.font_path.name, hex(location.codepoint))

        """
        idx = self._normalize_index(idx)
        font_path, face_idx, instance_idx, codepoint, style_idx, content_idx = (
            self._dataset.locate(idx)
        )
        return GlyphLocation(
            font_path=Path(font_path),
            face_idx=int(face_idx),
            instance_idx=None if instance_idx is None else int(instance_idx),
            codepoint=int(codepoint),
            style_idx=int(style_idx),
            content_idx=int(content_idx),
        )

    @property
    def targets(self) -> Tensor:
        """Label matrix pairing every sample with its style and content class.

        Returns:
            torch.LongTensor: Tensor of shape ``(N, 2)`` where column 0 holds
            the style class index and column 1 holds the content class index.

        Examples:
            >>> dataset = GlyphDataset(
            ...     root="fonts",
            ...     codepoints=range(0x41, 0x44),
            ... )
            >>> dataset.targets.shape
            torch.Size([N, 2])
            >>> dataset.targets[0]
            tensor([style_idx, content_idx])

        """
        raw = self._dataset.targets()
        if not raw:
            return torch.empty(0, 2, dtype=torch.long)
        return torch.frombuffer(bytearray(raw), dtype=torch.long).view(-1, 2)

    @property
    def content_classes(self) -> list[str]:
        """List of unique characters (Unicode strings) in the dataset.

        Returns class names sorted by their index. Each name is a single
        Unicode character corresponding to a codepoint in the dataset.

        Returns:
            list[str]: Character strings for each content class.

        Examples:
            >>> dataset = GlyphDataset(
            ...     root="fonts",
            ...     codepoints=range(0x41, 0x44),
            ... )
            >>> dataset.content_classes
            ['A', 'B', 'C']

        """
        return [label.char for label in self.metadata.contents]

    @property
    def content_class_to_idx(self) -> dict[str, int]:
        """Mapping from character strings to content class indices.

        Returns:
            dict[str, int]: Dictionary mapping character to index.

        Examples:
            >>> dataset.content_class_to_idx['A']
            0

        """
        return {label.char: label.idx for label in self.metadata.contents}

    @property
    def metadata(self) -> DatasetMetadata:
        """Structured style/content metadata for this dataset."""
        if self._metadata is None:
            self._metadata = build_dataset_metadata(
                style_names=self.style_classes,
                content_codepoints=self._dataset.content_classes,
            )
        return self._metadata

    @property
    def content_labels(self) -> list[ContentLabel]:
        """Content label metadata with explicit IDs and Unicode codepoints.

        Returns:
            list[ContentLabel]: Metadata entries ordered by ``idx``.

        """
        return list(self.metadata.contents)

    @property
    def content_label_to_idx(self) -> dict[str, int]:
        """Mapping from content label IDs to content class indices.

        Returns:
            dict[str, int]: Dictionary mapping ``label_id`` to content index.

        """
        return dict(self.metadata.content_id_to_idx)

    @property
    def style_classes(self) -> list[str]:
        """List of style variation instance names in the dataset.

        Returns class names sorted by their index. For variable fonts, names
        come from the font's named instances. For static fonts, names are
        derived from the font's family and subfamily names.

        Returns:
            list[str]: Descriptive names for each style class.

        Examples:
            >>> dataset.style_classes[:3]
            ['Roboto Regular', 'Roboto Bold', 'Lato Regular']

        """
        return list(self._dataset.style_classes)

    @property
    def style_labels(self) -> list[StyleLabel]:
        """Style label metadata with explicit IDs.

        Style names are not guaranteed to be unique, so each entry also includes
        a collision-safe ``label_id``.

        Returns:
            list[StyleLabel]: Metadata entries ordered by ``idx``.

        """
        return list(self.metadata.styles)

    @property
    def style_label_to_idx(self) -> dict[str, int]:
        """Mapping from style label IDs to style class indices.

        Returns:
            dict[str, int]: Dictionary mapping ``label_id`` to style index.

        """
        return dict(self.metadata.style_id_to_idx)

    @property
    def style_name_to_idxs(self) -> dict[str, list[int]]:
        """Mapping from style names to all matching style indices.

        Returns:
            dict[str, list[int]]: Dictionary mapping style display name to a
            list of all style indices that share that name.

        """
        return {
            name: list(idxs) for name, idxs in self.metadata.style_name_to_idxs.items()
        }


__all__ = [
    "ContentLabel",
    "DatasetMetadata",
    "GlyphDataset",
    "GlyphLocation",
    "GlyphSample",
    "StyleLabel",
]
