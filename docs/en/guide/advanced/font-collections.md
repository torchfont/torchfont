# Font Collections

TrueType Collections (`.ttc`) and OpenType Collections (`.otc`) store multiple
font faces in one file. `GlyphDataset` indexes every face in a collection
automatically:

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(
    root="data/fonts",
    patterns=("**/*.ttc", "**/*.otc"),
    transform=LoadGlyph(),
)

for font in dataset.font_classes:
    print(font.path, font.ttc_index)
```

Each face receives its own `font_idx`. `FontRef.ttc_index` identifies the face
within the file, beginning at zero; it is also zero for an ordinary single-face
font. A collection face otherwise behaves like any other font in the dataset.

## Selecting one face directly

Use `FontRef` when the file and face index are already known:

```python
from torchfont import FontRef, GlyphRef
from torchfont.transforms import LoadGlyph

ref = GlyphRef(
    font=FontRef("data/fonts/example.ttc", ttc_index=2),
    glyph_id=36,
)
outline = LoadGlyph()(ref)
```

An OpenType Collection may also contain variable faces. Face selection and
variation-location selection are independent: `ttc_index` selects a face, then
`LoadGlyph(location="random")` samples that face's variation axes.

See [Variable Fonts](./variable-fonts.md) for location sampling and explicit
axis values.
