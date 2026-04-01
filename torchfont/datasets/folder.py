"""Utilities for turning local font folders into indexed glyph datasets.

Notes:
    Glyph data is cached inside the native backend for the lifetime of each
    dataset instance. Recreate the dataset when editing font files on disk to
    ensure changes are observed.

Examples:
    Iterate glyph samples from a directory of fonts::

        from torchfont.datasets import FontFolder

        dataset = FontFolder(root="~/fonts")
        sample = dataset[0]
        print(sample.types, sample.style_idx)

"""

from collections.abc import Callable, Sequence
from pathlib import Path
from typing import NamedTuple, SupportsIndex

import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont.io.outline import COORD_DIM
from torchfont.sample import GlyphSample


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


class FontFolder(Dataset[GlyphSample]):
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
        style_class_to_idx (dict[str, int]): Mapping from style names to style
            class indices. This is a legacy convenience mapping and may collapse
            duplicate style names.
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

    See Also:
        torchfont.datasets.repo.FontRepo: Extends the same indexing machinery
        with Git synchronization for remote repositories.

    """

    def __init__(
        self,
        root: Path | str,
        *,
        codepoint_filter: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: (Callable[[GlyphSample], GlyphSample] | None) = None,
    ) -> None:
        """Initialize the dataset by scanning font files and indexing samples.

        Args:
            root (Path | str): Directory containing font files. Both OTF and TTF
                files are discovered recursively.
            codepoint_filter (Sequence[SupportsIndex] | None): Optional iterable
                of Unicode code points used to restrict the dataset content.
            patterns (Sequence[str] | None): Optional gitignore-style patterns
                describing which font paths to include.
            transform (Callable[[GlyphSample], GlyphSample] | None):
                Optional transformation applied to each sample before the item
                is returned.

        Examples:
            Restrict the dataset to uppercase ASCII glyphs::

                dataset = FontFolder(
                    root="~/fonts",
                    codepoint_filter=[ord(c) for c in "ABCDEFGHIJKLMNOPQRSTUVWXYZ"],
                )

        """
        self.root = Path(root).expanduser().resolve()
        self.transform = transform
        self.patterns = (
            tuple(str(pattern) for pattern in patterns)
            if patterns is not None
            else None
        )
        self.codepoint_filter = (
            [int(cp) for cp in codepoint_filter]
            if codepoint_filter is not None
            else None
        )

        self._dataset = _torchfont.FontDataset(
            str(self.root),
            self.codepoint_filter,
            self.patterns,
        )

    def __repr__(self) -> str:
        """Return a human-readable summary of this dataset.

        Returns:
            str: String showing the class name, root path, sample count,
                number of style classes, and number of content classes.

        Examples:
            >>> ds = FontFolder(
            ...     root="tests/fonts",
            ...     codepoint_filter=range(0x41, 0x5B),
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
        self._dataset = _torchfont.FontDataset(
            str(self.root),
            self.codepoint_filter,
            self.patterns,
        )

    def __len__(self) -> int:
        """Return the total number of glyph samples discoverable in the dataset.

        Returns:
            int: Total number of glyph samples available in the dataset.

        """
        return int(self._dataset.sample_count)

    def __getitem__(self, idx: int) -> GlyphSample:
        """Load a glyph sample and its associated targets.

        Args:
            idx (int): Zero-based index locating a sample across all fonts, code
                points, and instances. Negative indices are supported and count
                from the end of the dataset.

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
        idx = int(idx)
        original_idx = idx
        dataset_len = len(self)
        if idx < 0:
            idx += dataset_len
        if idx < 0 or idx >= dataset_len:
            msg = (
                f"index {original_idx} is out of range for dataset of length "
                f"{dataset_len}"
            )
            raise IndexError(msg)
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

    @property
    def targets(self) -> Tensor:
        """Label matrix pairing every sample with its style and content class.

        Returns:
            torch.LongTensor: Tensor of shape ``(N, 2)`` where column 0 holds
            the style class index and column 1 holds the content class index.

        Examples:
            >>> dataset = FontFolder(root="fonts", codepoint_filter=range(0x41, 0x44))
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
            >>> dataset = FontFolder(root="fonts", codepoint_filter=range(0x41, 0x44))
            >>> dataset.content_classes
            ['A', 'B', 'C']

        """
        codepoints = self._dataset.content_classes
        return [chr(cp) for cp in codepoints]

    @property
    def content_class_to_idx(self) -> dict[str, int]:
        """Mapping from character strings to content class indices.

        Returns:
            dict[str, int]: Dictionary mapping character to index.

        Examples:
            >>> dataset.content_class_to_idx['A']
            0

        """
        return {label.char: label.idx for label in self.content_labels}

    @property
    def content_labels(self) -> list[ContentLabel]:
        """Content label metadata with explicit IDs and Unicode codepoints.

        Returns:
            list[ContentLabel]: Metadata entries ordered by ``idx``.

        """
        codepoints = self._dataset.content_classes
        return [
            ContentLabel(
                idx=idx,
                label_id=f"content:U+{cp:04X}",
                char=chr(cp),
                codepoint=cp,
            )
            for idx, cp in enumerate(codepoints)
        ]

    @property
    def content_label_to_idx(self) -> dict[str, int]:
        """Mapping from content label IDs to content class indices.

        Returns:
            dict[str, int]: Dictionary mapping ``label_id`` to content index.

        """
        return {label.label_id: label.idx for label in self.content_labels}

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
    def style_class_to_idx(self) -> dict[str, int]:
        """Mapping from style instance names to style class indices.

        Returns:
            dict[str, int]: Dictionary mapping style name to index.

        Examples:
            >>> dataset.style_class_to_idx['Roboto Regular']
            0

        """
        return {name: idxs[-1] for name, idxs in self.style_name_to_idxs.items()}

    @property
    def style_labels(self) -> list[StyleLabel]:
        """Style label metadata with explicit IDs.

        Style names are not guaranteed to be unique, so each entry also includes
        a collision-safe ``label_id``.

        Returns:
            list[StyleLabel]: Metadata entries ordered by ``idx``.

        """
        return [
            StyleLabel(idx=idx, label_id=f"style:{idx}", name=name)
            for idx, name in enumerate(self.style_classes)
        ]

    @property
    def style_label_to_idx(self) -> dict[str, int]:
        """Mapping from style label IDs to style class indices.

        Returns:
            dict[str, int]: Dictionary mapping ``label_id`` to style index.

        """
        return {label.label_id: label.idx for label in self.style_labels}

    @property
    def style_name_to_idxs(self) -> dict[str, list[int]]:
        """Mapping from style names to all matching style indices.

        Returns:
            dict[str, list[int]]: Dictionary mapping style display name to a
            list of all style indices that share that name.

        """
        grouped: dict[str, list[int]] = {}
        for label in self.style_labels:
            grouped.setdefault(label.name, []).append(label.idx)
        return grouped
