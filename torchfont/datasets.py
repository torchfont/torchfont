"""Utilities for turning local font folders into indexed glyph datasets.

Notes:
    Native indexing state is built for the lifetime of each dataset instance.
    Keep that state in sync with the font files on disk by recreating the
    dataset after editing font files. If files change while a dataset instance
    is still in use, results are undefined and may include incorrect samples
    or runtime errors.

    Unpickling (including ``DataLoader`` worker spawn) rebuilds the index from
    the files on disk and verifies it against the pickled structure
    fingerprint, raising ``RuntimeError`` instead of silently misaligning
    labels when paths, faces, code points, or variation instance counts changed
    in between. Outline and metadata-only edits that preserve this structure
    are intentionally read from the current files on disk.

Examples:
    Iterate glyph samples from a directory of fonts::

        from torchfont.datasets import GlyphDataset

        dataset = GlyphDataset(root="~/fonts")
        sample = dataset[0]
        print(sample.types, sample.style_idx)

"""

import dataclasses
from collections.abc import Callable, Sequence
from operator import index
from pathlib import Path
from typing import Generic, SupportsIndex, TypeVar, cast, overload

import torch
from torch import Tensor
from torch.utils.data import Dataset

from torchfont import _torchfont
from torchfont.io import COORD_DIM
from torchfont.metadata import (
    DatasetMetadata,
    StyleAxis,
    build_dataset_metadata,
)

_T = TypeVar("_T")


@dataclasses.dataclass(frozen=True)
class NameRecord:
    """Strings from a font's ``name`` table, one field per NameID (0-25).

    Each field maps to exactly one NameID and is an empty string when
    the entry is absent in the font.
    """

    copyright_notice: str  # ID 0
    family_name: str  # ID 1
    subfamily_name: str  # ID 2
    unique_id: str  # ID 3
    full_name: str  # ID 4
    version_string: str  # ID 5
    postscript_name: str  # ID 6
    trademark: str  # ID 7
    manufacturer: str  # ID 8
    designer: str  # ID 9
    description: str  # ID 10
    vendor_url: str  # ID 11
    designer_url: str  # ID 12
    license_description: str  # ID 13
    license_url: str  # ID 14
    reserved: str  # ID 15
    typographic_family_name: str  # ID 16
    typographic_subfamily_name: str  # ID 17
    compatible_full_name: str  # ID 18
    sample_text: str  # ID 19
    postscript_cid_name: str  # ID 20
    wws_family_name: str  # ID 21
    wws_subfamily_name: str  # ID 22
    light_background_palette: str  # ID 23
    dark_background_palette: str  # ID 24
    variations_postscript_name_prefix: str  # ID 25


@dataclasses.dataclass
class GlyphSample:
    """One glyph sample returned by a dataset.

    Attributes:
        types (Tensor): 1-D long tensor of element types.
        coords (Tensor): 2-D float tensor of shape ``(N, 6)`` holding the
            coordinates for each path element.
        style_idx (int): Index into the dataset's ``style_classes`` list.
        content_idx (int): Index into the dataset's ``content_classes`` list.
        head (Tensor): ``head`` table fields (8,): ``units_per_em``, ``flags``,
            ``x_min``, ``y_min``, ``x_max``, ``y_max``, ``mac_style``,
            ``lowest_rec_ppem``. Bounding box values are in em units (font
            design units divided by ``unitsPerEm``).
        hhea (Tensor): ``hhea`` table fields (10,): ``ascender``,
            ``descender``, ``line_gap``, ``advance_width_max``, ``min_lsb``,
            ``min_rsb``, ``x_max_extent``, ``caret_slope_rise``,
            ``caret_slope_run``, ``caret_offset``. Metric lengths are
            in em units; ``caret_slope_rise`` and ``caret_slope_run`` are
            raw integers (dimensionless slope, not a length).
        os2 (Tensor): ``OS/2`` table fields (42,): ``weight_class``,
            ``width_class``, ``fs_type``, ``fs_selection``, typo/win metrics
            (6: ``typo_ascender``, ``typo_descender``, ``typo_line_gap``,
            ``win_ascent``, ``win_descent``, ``avg_char_width``),
            subscript/superscript (8), strikeout (2),
            ``s_family_class``, panose (10), vend_id (4), first/last char
            index (2), ``x_height``, ``cap_height``, ``default_char``,
            ``break_char``, ``max_context``. Metric values are in em units;
            ``nan`` when absent. Note: ``win_descent`` is stored as a
            **positive** value (matching the OpenType spec's unsigned
            ``usWinDescent``), whereas ``typo_descender`` and
            ``hhea.descender`` are **negative**.
        post (Tensor): ``post`` table fields (4,): ``italic_angle``,
            ``is_fixed_pitch`` (0.0 or 1.0), ``underline_position``,
            ``underline_thickness``. ``italic_angle`` is stored in degrees;
            underline metrics are in em units.
        maxp (Tensor): ``maxp`` table fields (14,) starting with
            ``num_glyphs``. TrueType-only fields are ``nan`` for CFF fonts.
        hmtx (Tensor): ``advance_width``, ``lsb`` (2,). Values are in em units.
        bounds (Tensor): ``x_min``, ``y_min``, ``x_max``, ``y_max`` (4,).
            Values are in em units.
        name (NameRecord): Strings from the ``name`` table, one field
            per NameID 0-25. Each field is an empty string when absent.
        codepoint (int): Unicode code point of the glyph (e.g. ``0x0041`` for 'A').
        glyph_name (str): PostScript name of the glyph.

    """

    types: Tensor
    coords: Tensor
    style_idx: int
    content_idx: int
    head: Tensor
    hhea: Tensor
    os2: Tensor
    post: Tensor
    maxp: Tensor
    hmtx: Tensor
    bounds: Tensor
    name: NameRecord
    codepoint: int
    glyph_name: str


class GlyphDataset(Dataset[_T], Generic[_T]):
    """Dataset that yields glyph samples or transform outputs from font files.

    The dataset flattens every available code point and variation instance into
    a single indexable sequence. By default, each item returns the loader output
    along with style and content targets as a ``GlyphSample``. When ``transform``
    is provided, each item returns the transform output instead.

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
        root (Path): Resolved root directory used for font discovery.
        patterns (tuple[str, ...] | None): Canonicalized path filter patterns.
        codepoints (tuple[int, ...] | None): Sorted unique Unicode code points
            applied during indexing.

    """

    @overload
    def __init__(
        self: "GlyphDataset[GlyphSample]",
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: None = None,
    ) -> None: ...

    @overload
    def __init__(
        self: "GlyphDataset[_T]",
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: Callable[[GlyphSample], _T],
    ) -> None: ...

    def __init__(
        self,
        root: Path | str,
        *,
        codepoints: Sequence[SupportsIndex] | None = None,
        patterns: Sequence[str] | None = None,
        transform: (Callable[[GlyphSample], _T] | None) = None,
    ) -> None:
        """Initialize the dataset by scanning font files and indexing samples.

        Args:
            root (Path | str): Directory containing font files. Both OTF and TTF
                files are discovered recursively.
            codepoints (Sequence[SupportsIndex] | None): Optional iterable
                of Unicode code points used to restrict the dataset content.
                Duplicate values are ignored and the effective filter is stored
                as sorted unique integers on ``dataset.codepoints``.
                Values that do not appear in any font charmap (including
                surrogates or values outside the Unicode range) simply
                produce no samples. Negative values will fail during
                conversion to the native unsigned integer type.
            patterns (Sequence[str] | None): Optional gitignore-style patterns
                describing which font paths to include. No implicit filtering
                from hidden directories or ignore files (such as ``.gitignore``,
                ``.ignore``, global gitignore, or git exclude rules) is applied;
                all such behavior must be expressed via ``patterns``. VCS metadata
                directories such as ``.git`` remain excluded.
            transform (Callable[[GlyphSample], _T] | None): Optional
                transformation applied to each sample before the item is returned.
                When provided, ``_T`` is inferred from the transform return type
                and becomes the dataset item type.

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

        self._backend = _torchfont.GlyphDatasetBackend(
            str(self.root),
            self.codepoints,
            self.patterns,
        )
        self._fingerprint: int = self._backend.fingerprint

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
            f"styles={self._backend.style_class_count}, "
            f"content_classes={self._backend.content_class_count})"
        )

    def __len__(self) -> int:
        """Return the total number of glyph samples discoverable in the dataset.

        Returns:
            int: Total number of glyph samples available in the dataset.

        """
        return int(self._backend.sample_count)

    @overload
    def __getitem__(
        self: "GlyphDataset[GlyphSample]",
        idx: SupportsIndex,
    ) -> GlyphSample: ...

    @overload
    def __getitem__(self, idx: SupportsIndex) -> _T: ...

    def __getitem__(self, idx: SupportsIndex) -> _T:
        """Load a glyph sample and its associated targets.

        Args:
            idx (SupportsIndex): Zero-based index locating a sample across all
                fonts, code points, and instances. Any object implementing
                ``__index__`` is accepted. Negative indices are supported and
                count from the end of the dataset.

        Returns:
            _T: A ``GlyphSample`` when no transform is provided; otherwise the
            value returned by ``transform``.

        Examples:
            Retrieve the first glyph sample and its target labels::

                sample = dataset[0]
                print(sample.types, sample.style_idx)

            Retrieve the last glyph sample::

                sample = dataset[-1]

        """
        idx = self._normalize_index(idx)
        item = self._backend.item(idx)
        types = torch.from_numpy(item.types)
        coords = torch.from_numpy(item.coords).view(-1, COORD_DIM)
        sample = GlyphSample(
            types=types,
            coords=coords,
            style_idx=item.style_idx,
            content_idx=item.content_idx,
            head=torch.from_numpy(item.head),
            hhea=torch.from_numpy(item.hhea),
            os2=torch.from_numpy(item.os2),
            post=torch.from_numpy(item.post),
            maxp=torch.from_numpy(item.maxp),
            hmtx=torch.from_numpy(item.hmtx),
            bounds=torch.from_numpy(item.bounds),
            name=NameRecord(**item.name),
            codepoint=item.codepoint,
            glyph_name=item.glyph_name,
        )
        if self.transform is not None:
            return self.transform(sample)
        return cast("_T", sample)

    def __getstate__(self) -> dict[str, object]:
        """Return state without the native backend for worker reconstruction."""
        state = self.__dict__.copy()
        state.pop("_backend", None)
        return state

    def __setstate__(self, state: dict[str, object]) -> None:
        """Restore state and recreate the native backend after unpickling.

        The backend is rebuilt by re-scanning ``root``, so the restored index
        is verified against the pickled structure fingerprint. The fingerprint
        covers everything that determines the sample-to-label mapping: font
        file order, paths, face indices, code points, and variation instance
        counts.

        Outline and metadata-only edits are deliberately not fingerprinted:
        files on disk remain the source of truth when the sample-to-label
        structure is unchanged.

        Raises:
            RuntimeError: If the font files under ``root`` no longer produce
                the same index structure as when the dataset was pickled,
                which would silently misalign ``targets`` and class indices.

        """
        self.__dict__.update(state)
        self._validate_root_dir(self.root)
        backend = _torchfont.GlyphDatasetBackend(
            str(self.root),
            self.codepoints,
            self.patterns,
        )
        if backend.fingerprint != self._fingerprint:
            msg = (
                f"font files under {str(self.root)!r} no longer have the same "
                "dataset structure as when this object was pickled; sample "
                "indices and targets would be inconsistent. Recreate the "
                "dataset from the current files."
            )
            raise RuntimeError(msg)
        self._backend = backend

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

    def _style_axes(self) -> list[tuple[StyleAxis, ...]]:
        """Return style axis metadata aligned with ``style_classes`` order."""
        return [
            tuple(StyleAxis(tag=tag, value=float(value)) for tag, value in axes)
            for axes in self._backend.style_axes
        ]

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
        arr = self._backend.targets()
        if arr.size == 0:
            return torch.empty(0, 2, dtype=torch.long)
        return torch.from_numpy(arr).view(-1, 2)

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
        return [char for _, char, _ in self._backend.content_metadata_rows()]

    @property
    def content_class_to_idx(self) -> dict[str, int]:
        """Mapping from character strings to content class indices.

        Returns:
            dict[str, int]: Dictionary mapping character to index.

        Examples:
            >>> dataset.content_class_to_idx['A']
            0

        """
        return {
            char: idx
            for idx, (_, char, _) in enumerate(self._backend.content_metadata_rows())
        }

    @property
    def metadata(self) -> DatasetMetadata:
        """Structured style/content metadata for this dataset."""
        style_meta_rows = self._backend.style_metadata_rows()
        style_axes = self._style_axes()
        style_rows = [
            (name, label_id, axes)
            for (name, label_id), axes in zip(style_meta_rows, style_axes, strict=True)
        ]
        return build_dataset_metadata(
            style_rows=style_rows,
            content_rows=self._backend.content_metadata_rows(),
        )

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
        return [name for name, _ in self._backend.style_metadata_rows()]
