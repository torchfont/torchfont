# Dataset API

<!-- markdownlint-disable MD013 -->

`GlyphDataset` is the primary public dataset API in `torchfont.datasets`.

## GlyphSample

```python
from torchfont.datasets import GlyphSample
```

Default return type for `GlyphDataset.__getitem__`, and the input sample type
for dataset transforms and `torchfont.transforms`.

## DatasetMetadata

```python
from torchfont.datasets import DatasetMetadata
```

Structured label metadata returned by `GlyphDataset.metadata`.

## GlyphDataset

```python
from torchfont.datasets import GlyphDataset
from torchfont.variation import (
    DefaultInstantiation,
    GridInstantiation,
    NamedInstantiation,
    VariationInstantiation,
)
```

### Constructor (`GlyphDataset`)

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    variation: VariationInstantiation | None = None,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

Variation instantiation classes are exported from `torchfont.variation`.

`T` is the transform return type and therefore the dataset item type.

| Parameter          | Type                              | Description                         |
| ------------------ | --------------------------------- | ----------------------------------- |
| `root`             | `Path \| str`                     | root directory for font discovery   |
| `codepoints` | `Sequence[SupportsIndex] \| None` | restrict indexed Unicode codepoints |
| `patterns`         | `Sequence[str] \| None`           | gitignore-style path filtering      |
| `variation` | `VariationInstantiation \| None` | variable-font instantiation policy |
| `transform`        | `Callable \| None`                | sample-first preprocessing (`GlyphSample -> T`) |

### Behavior

- supported extensions: `.ttf` / `.otf` / `.ttc` / `.otc`
- `root` is resolved to an absolute `Path` during initialization
- `codepoints` are normalized to sorted unique integers before indexing
- static fonts are always indexed as one static style; variable fonts use
  `variation` in `fvar` user coordinate space
- `variation=None` is equivalent to `NamedInstantiation()`;
  `DefaultInstantiation()` uses the `fvar` default location;
  `NamedInstantiation()` uses named instances when present and otherwise falls
  back to the default location
- `GridInstantiation(axes={"wght": 7, "wdth": 3})` samples each listed axis on
  an evenly spaced grid (the given number of points, from axis minimum to
  maximum) and takes the Cartesian product across those axes; axes that are not
  listed are pinned to their `fvar` default. The instance count is the product
  of the listed point counts and is therefore independent of the font's total
  axis count. `axes` must list at least one axis, and every point count must be
  greater than zero
- no implicit ignore rules are applied (hidden directories, `.gitignore`,
  `.ignore`, global gitignore, and git exclude files are all ignored for
  discovery); use `patterns` for path selection
- VCS metadata directories such as `.git`, `.hg`, and `.svn` stay excluded
- `__getitem__` supports negative indices (`dataset[-1]`)
- out-of-range index raises `IndexError`

### Stored configuration

- `dataset.root`: resolved root `Path`
- `dataset.patterns`: tuple of path-filter patterns, or `None`
- `dataset.codepoints`: tuple of sorted unique codepoints, or `None`
- `dataset.variation`: instantiation policy as passed, or `None`

### Return value

```python
sample = dataset[idx]
```

| Field                | Type                | Shape          |
| -------------------- | ------------------- | -------------- |
| `sample.types`       | `torch.LongTensor`  | `(seq_len,)`   |
| `sample.coords`      | `torch.FloatTensor` | `(seq_len, 6)` |
| `sample.style_idx`   | `int`               | scalar         |
| `sample.content_idx` | `int`               | scalar         |
| `sample.head`        | `torch.FloatTensor` | `(8,)`         |
| `sample.hhea`        | `torch.FloatTensor` | `(10,)`        |
| `sample.os2`         | `torch.FloatTensor` | `(42,)`        |
| `sample.post`        | `torch.FloatTensor` | `(4,)`         |
| `sample.maxp`        | `torch.FloatTensor` | `(14,)`        |
| `sample.hmtx`        | `torch.FloatTensor` | `(2,)`         |
| `sample.bounds`      | `torch.FloatTensor` | `(4,)`         |
| `sample.name`        | `NameRecord`        | —              |
| `sample.codepoint`   | `int`               | —              |
| `sample.glyph_name`  | `str`               | —              |

`sample.post` stores `(italic_angle, is_fixed_pitch, underline_position,
underline_thickness)`. The italic angle is in degrees, `is_fixed_pitch` is
`0.0` or `1.0`, and the two underline metrics are in em units.

Without `transform`, `sample` is a `GlyphSample`. With `transform`, the
dataset item type is inferred from the transform return type.

### Properties

#### `targets -> torch.LongTensor`

Label matrix for all samples (`shape=(N, 2)`).

- `targets[:, 0]`: style index
- `targets[:, 1]`: content index

#### `content_classes -> list[str]`

Content class names (single-character Unicode strings).

#### `metadata -> DatasetMetadata`

Structured label metadata object.

- `metadata.styles`: tuple of `StyleLabel`
- `metadata.contents`: tuple of `ContentLabel`
- `metadata.style_id_to_idx`: mapping from style `label_id` to style index
- `metadata.style_name_to_idxs`: mapping from style display name to all indices
- `metadata.content_id_to_idx`: mapping from content `label_id` to content index

#### `content_class_to_idx -> dict[str, int]`

Mapping from character to content index.

#### `style_classes -> list[str]`

Style class names. Static fonts use family/subfamily names. Variable fonts use
the family name plus the selected variation location, such as
`Roboto wght=400,wdth=100`.

### Example (`GlyphDataset`)

```python
dataset = GlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),  # A-Z
    patterns=("**/*.ttf", "!*Bold*"),
)
```

## Choosing `root`

- a normal local font directory
- a repository checkout you cloned yourself with Git or another tool
- a cached copy of an external corpus managed outside TorchFont

TorchFont treats all of those as ordinary directories. If files on disk change,
recreate the dataset instance so the native indexing state stays in sync. If
files change while a dataset instance is still in use, results are undefined
and may include incorrect samples or runtime errors.
