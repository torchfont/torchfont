# Dataset API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` exposes reference-first PyTorch datasets. Dataset items are
small, pickle-friendly dataclasses. Pass `LoadGlyph()` or a composed transform
pipeline to the dataset's `transform` argument to load outlines lazily (see
[Transform Utilities](./transforms.md)).

Dataset indices and class targets are built from font files at construction
time. Glyph outlines and registered-axis values are loaded lazily from the
current files on disk. Modifying font files during a dataset object's lifetime,
including across pickle/unpickle boundaries, is unsupported and may produce
inconsistent samples or labels.

## Reference Types

```python
from torchfont.structures import (
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
| `GlyphSample` | `ref: GlyphRef`, `font_idx: int`, `style_idx: int`, `character_idx: int`, `weight: float \| None`, `width: float \| None`, `italic: float \| None`, `slant: float \| None`, `optical_size: float \| None` |
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
- `weight_targets -> FloatTensor (N,)`
- `width_targets -> FloatTensor (N,)`
- `italic_targets -> FloatTensor (N,)`
- `slant_targets -> FloatTensor (N,)`
- `optical_size_targets -> FloatTensor (N,)`

These targets use the registered OpenType user scales: weight is comparable
with CSS weight, width is a percentage, italic ranges from `0` (Roman) to `1`
(fully italic), slant is in degrees, and optical size is in points. All five
targets are floating point, including `italic_targets`, whose intermediate
variation coordinates are preserved. Each axis uses the indexed variation
location first. For axes absent from `fvar`, the fallbacks are
OS/2 `usWeightClass` for `wght`, OS/2 `usWidthClass` for `wdth`, OS/2
`fsSelection.ITALIC` (or `head.macStyle.ITALIC` when OS/2 is unavailable) for
`ital`, and `post.italicAngle` for `slnt`. `head.macStyle.BOLD` is not a weight
class and is therefore not converted to an arbitrary `wght` value. A value that
cannot be derived from the font is `NaN`; use `torch.isfinite` directly if a
loss needs to ignore unavailable targets.
The same values are available as `sample.weight`, `sample.width`,
`sample.italic`, `sample.slant`, and `sample.optical_size`, which is convenient
inside a transform. Unavailable sample values use `None` (target Tensors use
`NaN`).
The font files are parsed when one of these target properties is accessed; the
expanded target vectors are neither built at dataset construction nor cached.
OS/2 optical-size ranges are not collapsed to an arbitrary midpoint: `opsz`
is present only when the indexed variation location provides an actual
coordinate.

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
The instance-count function gives each font a discrete multiplicity without fixing concrete
locations. Static fonts are included as normal fonts.

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
    grid_instance_count,
    grid_instances,
    named_instance_count,
    named_instances,
)
```

- `named_instances(font)`: deduplicated fvar named instances, falling back to default
- `default_instance(font)`: one default location
- `grid_instances({"wght": 7, "wdth": 3})`: evenly spaced fixed grid
- `grid_instances({})`: one default location, the empty-grid identity
- `named_instance_count(font)`: count matching `named_instances`
- `default_instance_count(font)`: one instance slot
- `grid_instance_count({"wght": 7, "wdth": 3})`: count matching `grid_instances`

Grid policies pin unlisted axes to their defaults and ignore requested axes that
a particular font does not have, so one policy can cover heterogeneous font
collections.

For transform-time variation sampling, see `RandomLocation` in
[Transform Utilities](./transforms.md). Datasets do not have a dataset-level seed.

Custom instance functions may return zero locations. Unknown axes returned by a
custom function and duplicate locations after normalization raise
`ValueError` during dataset construction.
