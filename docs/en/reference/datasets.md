# Datasets

TorchFont uses one deterministic indexing rule: one dataset element for every
font face and supported Unicode codepoint. A variable font is one face, regardless
of its axes or named instances. Variation locations are selected by transforms,
not by expanding the dataset index.

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
| `GlyphRef` | `font: FontRef`, `codepoint: int` |
| `GlyphSample` | `ref: GlyphRef`, `font_idx: int`, `character_idx: int` |

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
at its default location, or `RandomLocation()` to draw one location whenever a
sample is transformed. On a static face, both transforms use the same empty
location.

```python
from torchfont.transforms import LoadGlyph, RandomLocation

evaluation = GlyphDataset(root, transform=LoadGlyph())
training = GlyphDataset(root, transform=RandomLocation())
```

Properties:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`
- `character_targets -> LongTensor (N,)`

The face-based rule is deliberately mechanical rather than statistically fair:
a family distributed as several static faces has more elements than the same
design distributed as one variable face. TorchFont does not infer families or a
measure over design space. Adjust training weights with a PyTorch sampler when
the application requires a different distribution.

## Loading explicit locations

The functional API remains available for deterministic replay:

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)  # face default
outline = F.load_glyph(sample.ref, {"wght": 700.0})
```
