# Datasets

TorchFont uses one deterministic indexing rule: one dataset element for every
font face and supported Unicode codepoint. Transforms select variation locations
when loading outlines.

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/google/fonts",
    codepoints=[0x41, 0x42],
    patterns="ofl/*/*.ttf",
)
```

Without a transform, `dataset[i]` returns a pickle-friendly `GlyphSample`:

| Type | Fields |
|---|---|
| `FontRef` | `path: str`, `ttc_index: int` |
| `GlyphRef` | `font: FontRef`, `glyph_id: int` |
| `GlyphSample` | `ref: GlyphRef`, `codepoint: int`, `font_idx: int`, `character_idx: int` |

## `GlyphDataset`

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: str | Sequence[str] | None = None,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

The index and raw samples are deterministic. Use `LoadGlyph()` to load each face
at its default location, or set `location="random"` to draw one location whenever
a sample is transformed. On a static face, both policies use the same empty
location.

```python
from torchfont.transforms import LoadGlyph

evaluation = GlyphDataset(root, transform=LoadGlyph())
training = GlyphDataset(root, transform=LoadGlyph(location="random"))
```

Properties:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`
- `character_targets -> LongTensor (N,)`

The sampling distribution is proportional to the number of supported
codepoints in each face. Adjust training weights with a PyTorch sampler when the
application requires a different distribution.

## Loading explicit locations

The functional API remains available for deterministic replay:

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)  # face default
outline = F.load_glyph(sample.ref, {"wght": 700.0})
```
