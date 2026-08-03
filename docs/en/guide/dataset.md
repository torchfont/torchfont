# Building a GlyphDataset

Point `GlyphDataset` at a local font directory:

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
print(len(dataset.character_classes))
```

Each element represents one font face and one codepoint supported by that face.
Every face contributes one element per supported codepoint.

## Filtering files

`patterns` accepts one gitignore-style pattern or a sequence:

```python
dataset = GlyphDataset(
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
dataset = GlyphDataset(
    root="data/google/fonts",
    codepoints=range(0x41, 0x5B),
)
```

Duplicate codepoints are removed and the index is deterministic. Fonts that do
not contain any requested outline glyph are omitted.

## Choosing variation locations

Raw samples are deterministic. Load default locations for evaluation:

```python
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(root="data/google/fonts", transform=LoadGlyph())
```

Draw one location per access for training:

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    transform=LoadGlyph(location="random"),
)
```

Random locations follow PyTorch RNG seeding. On static faces this produces the
same empty location as the default policy.
