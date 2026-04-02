# Dataset API

<!-- markdownlint-disable MD013 -->

`GlyphDataset` is the primary public dataset API in `torchfont.datasets`.

## GlyphSample

```python
from torchfont.datasets import GlyphSample
```

Structured return type for `GlyphDataset.__getitem__` and the sample type used
throughout `torchfont.transforms`.

## GlyphLocation

```python
from torchfont.datasets import GlyphLocation
```

Structured source metadata returned by `GlyphDataset.locate(idx)`.

## DatasetMetadata

```python
from torchfont.datasets import DatasetMetadata
```

Structured label metadata returned by `GlyphDataset.metadata`.

## GlyphDataset

```python
from torchfont.datasets import GlyphDataset
```

### Constructor (`GlyphDataset`)

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[GlyphSample], GlyphSample] | None = None,
)
```

| Parameter          | Type                              | Description                         |
| ------------------ | --------------------------------- | ----------------------------------- |
| `root`             | `Path \| str`                     | root directory for font discovery   |
| `codepoints` | `Sequence[SupportsIndex] \| None` | restrict indexed Unicode codepoints |
| `patterns`         | `Sequence[str] \| None`           | gitignore-style path filtering      |
| `transform`        | `Callable \| None`                | sample-first preprocessing (`GlyphSample -> GlyphSample`) |

### Behavior

- supported extensions: `.ttf` / `.otf` / `.ttc` / `.otc`
- `root` is resolved to an absolute `Path` during initialization
- `codepoints` are normalized to sorted unique integers before indexing
- invalid `codepoints` values raise `ValueError`; accepted values must be
  Unicode scalar values (`0 <= cp <= 0x10FFFF`, excluding surrogates)
- `__getitem__` supports negative indices (`dataset[-1]`)
- out-of-range index raises `IndexError`

### Stored configuration

- `dataset.root`: resolved root `Path`
- `dataset.patterns`: tuple of path-filter patterns, or `None`
- `dataset.codepoints`: tuple of sorted unique codepoints, or `None`

### Return value

```python
sample = dataset[idx]
```

| Field         | Type                | Shape          |
| ------------- | ------------------- | -------------- |
| `sample.types`       | `torch.LongTensor`  | `(seq_len,)`   |
| `sample.coords`      | `torch.FloatTensor` | `(seq_len, 6)` |
| `sample.style_idx`   | `int`               | scalar         |
| `sample.content_idx` | `int`               | scalar         |

`sample` is a `GlyphSample`.

### Methods

#### `locate(idx) -> GlyphLocation`

Map one dataset index back to its source font location.

- `font_path`: resolved path to the font file
- `face_idx`: zero-based face index within the file
- `instance_idx`: zero-based named-instance index, or `None` for static fonts
- `codepoint`: Unicode codepoint for the glyph sample
- `style_idx`: style class index
- `content_idx`: content class index

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

#### `content_labels -> list[ContentLabel]`

Content label metadata entries. Each entry has:

- `idx`: content index
- `label_id`: collision-safe ID (`content:U+XXXX`)
- `char`: one-character Unicode string
- `codepoint`: Unicode codepoint (`int`)

#### `content_label_to_idx -> dict[str, int]`

Mapping from content `label_id` to content index.

#### `style_classes -> list[str]`

Style class names. Static fonts use family/subfamily names. Variable fonts use
named instances when available, otherwise they fall back to family/subfamily
(or family-only) names. If a named instance exists but its subfamily name is
empty, the family name is used.

#### `style_labels -> list[StyleLabel]`

Style label metadata entries. Each entry has:

- `idx`: style index
- `label_id`: collision-safe ID (`style:<idx>`)
- `name`: display name (may duplicate)

#### `style_label_to_idx -> dict[str, int]`

Mapping from style `label_id` to style index.

#### `style_name_to_idxs -> dict[str, list[int]]`

Mapping from style display name to all matching style indices.

The metadata-related properties above are convenience projections from
`dataset.metadata`.

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
recreate the dataset instance so the native cache is rebuilt.
