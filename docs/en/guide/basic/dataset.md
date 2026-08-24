# Building a Dataset

Point `CodepointDataset` at a local font directory:

```python
from torchfont.datasets import CodepointDataset

dataset = CodepointDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
print(len(dataset.character_classes))
```

Each element represents one font face and one codepoint supported by that face.
Every face contributes one element per supported codepoint.

## Filtering files

`patterns` accepts one gitignore-style pattern or a sequence:

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/*.ttf",
    ),
)
```

## Filtering characters

Use integer Unicode codepoints:

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    codepoints=range(0x41, 0x5B),
)
```

Duplicate codepoints are removed and the index is deterministic. Fonts that do
not contain any requested outline glyph are omitted.

## Filtering by sequence length

`max_length` keeps only glyphs whose outline is at most that many elements long:

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    max_length=128,
)
```

Fonts left without a fitting glyph are omitted.

## Reaching glyphs no codepoint maps to

`CodepointDataset` indexes what the `cmap` table maps, so ligatures, alternates, and
other glyphs an OpenType feature substitutes into never appear in it. Index
every glyph a face draws with `GlyphIdDataset` instead:

```python
from torchfont.datasets import GlyphIdDataset

dataset = GlyphIdDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
```

Each element represents one font face and one glyph id, in ascending order and
starting at `.notdef`. Glyph ids are face-local, so this dataset takes no
`codepoints` filter and its samples carry no character target. `patterns`,
`max_length`, `transform`, and `LoadGlyph` work exactly as they do above.

## Choosing variation locations

Raw samples are deterministic. Load default locations for evaluation:

```python
from torchfont.transforms import LoadGlyph

dataset = CodepointDataset(root="data/google/fonts", transform=LoadGlyph())
```

Draw one location per access for training:

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    transform=LoadGlyph(location="random"),
)
```

Random locations follow PyTorch RNG seeding. On static faces this produces the
same empty location as the default policy.

See [Variable Fonts](../advanced/variable-fonts.md) for explicit locations and
sampling behavior.
