# Dataset API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` exposes reference-first PyTorch datasets. Dataset items are
small, pickle-friendly dataclasses; outline loading happens explicitly in a
transform with `load_glyph` (see [Transform Utilities](./transforms.md)).

Dataset indices and targets are built from font files at construction time,
while glyph outlines are loaded lazily from the current files on disk. Modifying
font files during a dataset object's lifetime, including across pickle/unpickle
boundaries, is unsupported and may produce inconsistent samples or labels.

## Reference Types

```python
from torchfont.datasets import (
    FontRef,
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
)
```

| Type | Fields |
| ---- | ------ |
| `FontRef` | `path: str`, `ttc_index: int` |
| `GlyphRef` | `font: FontRef`, `codepoint: int`, `location: Mapping[str, float]` |
| `VariableGlyphRef` | `font: FontRef`, `codepoint: int` |
| `GlyphSample` | `ref: GlyphRef`, `font_idx: int`, `style_idx: int`, `character_idx: int` |
| `VariableGlyphSample` | `ref: VariableGlyphRef`, `font_idx: int`, `character_idx: int` |

`ttc_index` follows the name used internally by read-fonts/skrifa for the
font's index inside a TrueType Collection. For a single-font file it is `0`.

## GlyphDataset

```python
from torchfont.datasets import GlyphDataset
from torchfont.instance_fn import named_instances

dataset = GlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),
    patterns=("**/*.ttf",),
    instance_fn=named_instances,
)
```

`GlyphDataset` indexes fixed variation locations. The instance function runs only
at construction time and is not stored in pickle state. Without `transform`,
`dataset[i]` returns a `GlyphSample`.

Constructor:

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    instance_fn: InstanceLocationsFn = torchfont.instance_fn.named_instances,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

Targets:

- `font_targets -> LongTensor (N,)`
- `style_targets -> LongTensor (N,)`
- `character_targets -> LongTensor (N,)`

Class vocabularies:

- `font_classes -> list[FontRef]`
- `style_classes -> list[str]`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`

## VariableGlyphDataset

```python
from torchfont.datasets import VariableGlyphDataset
from torchfont.instance_fn import named_instance_count

dataset = VariableGlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),
    instance_fn=named_instance_count,
)
```

`VariableGlyphDataset` does not put a location in the index. Use it for training
augmentation where the transform samples a fresh location for each access.
`instance_fn` is an instance-count function: it gives each font a discrete multiplicity
without fixing concrete locations. Static fonts are included as normal fonts.

Constructor:

```python
VariableGlyphDataset(
    root: Path | str,
    *,
    instance_fn: InstanceCountFn = torchfont.instance_fn.named_instance_count,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[VariableGlyphSample], T] | None = None,
)
```

Targets:

- `font_targets -> LongTensor (N,)`
- `character_targets -> LongTensor (N,)`

## Instance Functions

```python
from torchfont.instance_fn import (
    default_instance,
    default_instance_count,
    grid_instances,
    grid_instance_count,
    named_instances,
    named_instance_count,
)
```

Built-ins:

- `named_instances(font)`: fvar named instances, deduplicated; falls back to default
- `default_instance(font)`: one default location
- `grid_instances({"wght": 7, "wdth": 3})`: evenly spaced fixed grid; axes absent from a font are ignored, unlisted axes use their defaults, and static fonts use one default instance
- `named_instance_count(font)`: instance count matching `named_instances`
- `default_instance_count(font)`: one instance slot
- `grid_instance_count({"wght": 7, "wdth": 3})`: instance count matching `grid_instances`

For transform-time variation sampling, see `random_location` in
[Transform Utilities](./transforms.md). Datasets do not have a dataset-level seed.

Custom instance functions may return zero locations. Unknown axes and duplicate
locations after normalization raise `ValueError` during dataset construction.
