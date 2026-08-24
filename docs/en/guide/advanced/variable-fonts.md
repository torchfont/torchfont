# Variable Fonts

A variable font represents a family of related designs with axes such as
weight (`wght`), width (`wdth`), and optical size (`opsz`). `LoadGlyph` resolves
an axis location while loading each glyph.

Use the default location for deterministic evaluation:

```python
from torchfont.datasets import CodepointDataset
from torchfont.transforms import LoadGlyph

evaluation = CodepointDataset(
    root="data/fonts",
    patterns=("**/*.ttf", "**/*.otf", "**/*.ttc", "**/*.otc"),
    transform=LoadGlyph(),
)
```

Set `location="random"` to sample every axis uniformly between its minimum and
maximum on each access. This makes the font's design space available as training
augmentation without creating separate files for the instances:

```python
training = CodepointDataset(
    root="data/fonts",
    transform=LoadGlyph(location="random"),
)

sample = training[0]
print(sample.location)  # For example: {"wght": 573.2, "opsz": 41.7}
```

The sampled values use PyTorch's random number generator, so DataLoader worker
seeding and `torch.manual_seed` apply. `GlyphData.location` records the actual
axis values used for the returned outline. On a static font it is an empty
dictionary, and the random policy produces the same outline as the default
policy.

`LoadGlyph(location="random")` samples all axes. Use
`functional.load_glyph(ref, location={...})` when an experiment requires an
explicit location:

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(
    sample.ref,
    location={"wght": 700.0, "wdth": 90.0},
)
```
