# Dataset API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` provides `torch.utils.data.Dataset`-compatible classes.

## Which class to use

| Class         | Source                   | Best for                              |
| ------------- | ------------------------ | ------------------------------------- |
| `FontFolder`  | local directory          | quick experiments with local fonts    |
| `FontRepo`    | arbitrary Git repository | pinned OSS font datasets by ref       |
| `GoogleFonts` | Google Fonts repository  | large-scale experiments with defaults |

## FontFolder

```python
from torchfont.datasets import FontFolder
```

### Constructor (`FontFolder`)

```python
FontFolder(
    root: Path | str,
    *,
    codepoint_filter: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[GlyphSample], GlyphSample] | None = None,
)
```

| Parameter          | Type                              | Description                         |
| ------------------ | --------------------------------- | ----------------------------------- |
| `root`             | `Path \| str`                     | root directory for font discovery   |
| `codepoint_filter` | `Sequence[SupportsIndex] \| None` | restrict indexed Unicode codepoints |
| `patterns`         | `Sequence[str] \| None`           | gitignore-style path filtering      |
| `transform`        | `Callable \| None`                | sample-first preprocessing (`GlyphSample -> GlyphSample`) |

### Behavior

- supported extensions: `.ttf` / `.otf` / `.ttc` / `.otc`
- `__getitem__` supports negative indices (`dataset[-1]`)
- out-of-range index raises `IndexError`

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

#### `style_class_to_idx -> dict[str, int]`

Legacy convenience mapping from style name to style index. If style names
collide, later entries overwrite earlier ones.

#### `style_labels -> list[StyleLabel]`

Style label metadata entries. Each entry has:

- `idx`: style index
- `label_id`: collision-safe ID (`style:<idx>`)
- `name`: display name (may duplicate)

#### `style_label_to_idx -> dict[str, int]`

Mapping from style `label_id` to style index.

#### `style_name_to_idxs -> dict[str, list[int]]`

Mapping from style display name to all matching style indices.

The legacy metadata-related properties above are projections from
`dataset.metadata`.

### Example (`FontFolder`)

```python
dataset = FontFolder(
    root="~/fonts",
    codepoint_filter=range(0x41, 0x5B),  # A-Z
    patterns=("**/*.ttf", "!*Bold*"),
)
```

## FontRepo

```python
from torchfont.datasets import FontRepo
```

### Constructor (`FontRepo`)

```python
FontRepo(
    root: Path | str,
    url: str,
    ref: str,
    *,
    patterns: Sequence[str],
    codepoint_filter: Sequence[SupportsIndex] | None = None,
    transform: Callable[[GlyphSample], GlyphSample] | None = None,
    download: bool = False,
    depth: int = 1,
)
```

| Parameter          | Type                              | Description                         |
| ------------------ | --------------------------------- | ----------------------------------- |
| `root`             | `Path \| str`                     | local path for the Git working tree |
| `url`              | `str`                             | remote repository URL               |
| `ref`              | `str`                             | Git ref (see Notes for `download=True` constraints) |
| `patterns`         | `Sequence[str]`                   | gitignore-style path filtering      |
| `codepoint_filter` | `Sequence[SupportsIndex] \| None` | codepoint restriction               |
| `transform`        | `Callable \| None`                | preprocessing transform             |
| `download`         | `bool`                            | fetch from remote when `True`       |
| `depth`            | `int`                             | libgit2 fetch depth (`1` shallow)   |

### Extra properties

| Property      | Type  | Description                      |
| ------------- | ----- | -------------------------------- |
| `url`         | `str` | URL argument stored on dataset   |
| `ref`         | `str` | Git ref passed to constructor    |
| `commit_hash` | `str` | checked-out commit hash          |

### Notes

- Git sync is implemented via `pygit2`/libgit2
- checkout uses a force strategy in both modes
- with `download=False`, unresolved local `ref` raises an exception
- `download=False` on a missing `root/.git` raises `FileNotFoundError`
- with `download=True`, remote-tracking refs (`origin/main`) and ref expressions
  (`main~1`, `HEAD^`, `a:b`) are rejected
- with `download=True`, branch shorthand fetches `refs/heads/<ref>`;
  use explicit `refs/tags/...` for tags
- if existing `root/.git` has different `origin` URL, `ValueError` is raised
- if existing `root/.git` has no `origin` remote, `ValueError` is raised

### Example (`FontRepo`)

```python
dataset = FontRepo(
    root="data/fortawesome/font-awesome",
    url="https://github.com/FortAwesome/Font-Awesome",
    ref="7.x",
    patterns=("otfs/*.otf",),
    download=True,
)
```

## GoogleFonts

```python
from torchfont.datasets import GoogleFonts
```

### Constructor (`GoogleFonts`)

```python
GoogleFonts(
    root: Path | str,
    ref: str,
    *,
    patterns: Sequence[str] | None = None,
    codepoint_filter: Sequence[int] | None = None,
    transform: Callable[[GlyphSample], GlyphSample] | None = None,
    download: bool = False,
    depth: int = 1,
)
```

### Defaults

When `patterns=None`:

```python
(
    "apache/*/*.ttf",
    "ofl/*/*.ttf",
    "ufl/*/*.ttf",
    "!ofl/adobeblank/AdobeBlank-Regular.ttf",
)
```

### Notes (`GoogleFonts`)

- source repository URL is fixed to `https://github.com/google/fonts`
- same Git sync behavior as `FontRepo` (`download`, `depth`, URL consistency checks)
- use a dedicated `root` directory for Google Fonts cache

### Example (`GoogleFonts`)

```python
dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    codepoint_filter=range(0x30, 0x3A),
    download=True,
)
```
