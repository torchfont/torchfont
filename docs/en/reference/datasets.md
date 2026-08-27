# Datasets

TorchFont indexes a local font directory two ways. `CodepointDataset` counts one
element for every font face and supported Unicode codepoint. `GlyphIdDataset`
counts one element for every font face and outline glyph, including glyphs no
codepoint maps to. Both indexes are deterministic, and transforms select
variation locations when loading outlines.

| Dataset | One element per | Sample | Filters |
|---|---|---|---|
| `CodepointDataset` | face and codepoint | `CodepointSample` | `codepoints`, `max_length`, `patterns` |
| `GlyphIdDataset` | face and glyph id | `GlyphIdSample` | `max_length`, `patterns` |

Without a transform, `dataset[i]` returns a pickle-friendly sample:

| Type | Fields |
|---|---|
| `FontRef` | `path: str`, `face_index: int` |
| `GlyphRef` | `font: FontRef`, `glyph_id: int` |
| `CodepointSample` | `ref: GlyphRef`, `codepoint: int`, `font_idx: int`, `character_idx: int` |
| `GlyphIdSample` | `ref: GlyphRef`, `font_idx: int` |

## `CodepointDataset`

```python
CodepointDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    max_length: SupportsIndex | None = None,
    patterns: str | Sequence[str] | None = None,
    transform: Callable[[CodepointSample], T] | None = None,
)
```

```python
from torchfont.datasets import CodepointDataset

dataset = CodepointDataset(
    root="data/google/fonts",
    codepoints=[0x41, 0x42],
    patterns="ofl/*/*.ttf",
)
```

Each element pairs one face with one codepoint its `cmap` maps to an outline
glyph. Ligatures, alternates, and other glyphs no codepoint reaches are not
part of this index.

Properties:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`
- `character_targets -> LongTensor (N,)`
- `outline_lengths -> LongTensor (N,)`

The sampling distribution is proportional to the number of supported
codepoints in each face. Adjust training weights with a PyTorch sampler when the
application requires a different distribution.

## `GlyphIdDataset`

```python
GlyphIdDataset(
    root: Path | str,
    *,
    max_length: SupportsIndex | None = None,
    patterns: str | Sequence[str] | None = None,
    transform: Callable[[GlyphIdSample], T] | None = None,
)
```

```python
from torchfont.datasets import GlyphIdDataset

dataset = GlyphIdDataset(
    root="data/google/fonts",
    patterns="ofl/*/*.ttf",
)
```

Each element pairs one face with one glyph it draws an outline for, in
ascending glyph id order and starting at `.notdef`. Ligatures, alternates, and
every other glyph an OpenType feature substitutes into are reachable here.

Glyph ids are face-local, so this dataset takes no `codepoints` filter and its
samples carry no codepoint or character target.

Properties:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `glyph_ids -> LongTensor (N,)`
- `outline_lengths -> LongTensor (N,)`

`glyph_ids` holds the face-local glyph id of each sample. Ids are not
comparable across faces, so they are data for selecting samples rather than
class indices to train against.

The sampling distribution is proportional to the number of outline glyphs in
each face.

## Filtering by sequence length

Both datasets take `max_length`. It keeps only glyphs whose outline is at most
that many elements long.

```python
dataset = GlyphIdDataset(root="data/google/fonts", max_length=128)
```

Faces left without a fitting glyph are omitted, as they are for the other
filters.

`outline_lengths` contains each pre-transform encoded length, including the
trailing `END` element, at the face's default variation location.

## Loading glyphs

The index and raw samples are deterministic. Use `LoadGlyph()` to load each face
at its default location, or set `location="random"` to draw one location whenever
a sample is transformed. On a static face, both policies use the same empty
location.

```python
from torchfont.transforms import LoadGlyph

evaluation = CodepointDataset(root, transform=LoadGlyph())
training = CodepointDataset(root, transform=LoadGlyph(location="random"))
```

`LoadGlyph` returns a `CodepointData` for a `CodepointSample` and a
`GlyphIdData` for a `GlyphIdSample`. Both carry the loaded payload, the glyph
reference, the resolved location, and the continuous targets; only
`CodepointData` carries the codepoint targets. See
[Core Types](./core-types.md).

## Loading explicit locations

The functional API remains available for deterministic replay:

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)  # face default
outline = F.load_glyph(sample.ref, {"wght": 700.0})
```
