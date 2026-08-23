# Glyph Data Format

## Accessing a sample

Access a sample from the dataset created in the previous chapter. Run the following code:

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
    transform=LoadGlyph(),
)

data = dataset[0]
outline = data.data
types, coords = outline.types, outline.coords

print(data.ref)  # glyph reference
print(types)  # element type sequence
print(coords)  # coordinates sequence
print(data.codepoint)  # Unicode codepoint
print(data.font_idx)  # font face class ID
print(data.character_idx)  # character class ID
print(data.weight)  # OpenType weight
print(data.width)  # OpenType width percentage
print(data.italic)  # OpenType italic value
print(data.slant)  # slant angle
print(data.optical_size)  # optical size in points
```

The return value is `GlyphData[Outline]`. It keeps the semantic outline,
deterministic glyph reference, and dataset-local targets in one shallow record.
Indices are Python integers and continuous targets are floats. A continuous
target unavailable in the font is `None`. A training application's local
`collate_fn` can convert selected targets to tensors and choose its own missing
value representation, such as `NaN` plus a mask.

## Outline model

A glyph outline is represented as a sequence of path elements.

- **Path element**: the smallest unit, consisting of one element type and one coordinates row
- **Subpath**: a sequence of path elements representing one continuous curve that makes up a glyph
- **Outline**: a sequence of path elements representing the contour of one glyph

`types` is a `(seq_len,)` tensor with dtype `torch.long`. `coords` is a
`(seq_len, 6)` tensor with dtype `torch.float32` for outlines returned by
`LoadGlyph`.

## Element type

Element types are defined in `ElementType`. Run the following code to see the mapping between values and names:

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph
from torchfont import ElementType

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
    transform=LoadGlyph(),
)

outline = dataset[0].data
types, coords = outline.types, outline.coords

print(types)
print(ElementType(types[0].item()).name)
```

You will see output like:

```
tensor([1, 2, 3, ..., 5, 6])
MOVE_TO
```

The seven types are `MOVE_TO`, `LINE_TO`, `QUAD_TO`, `CURVE_TO`, `CLOSE`,
`END`, and `PAD`.

- `ElementType.END` marks the end of the sequence
- `ElementType.PAD` marks rows introduced by padding a batch. Loading a font
  never produces it, and a single glyph never contains it. Read
  `outline.padding_mask` rather than comparing against the value.

## Coordinates

Each path element uses a 6D coordinates vector. Run the following code to inspect the shape and contents:

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
    transform=LoadGlyph(),
)

outline = dataset[0].data
types, coords = outline.types, outline.coords

print(coords.shape)
print(coords[0])
```

You will see output like:

```
torch.Size([seq_len, 6])
tensor([cx0, cy0, cx1, cy1, x, y])
```

Which dimensions are used depends on the element type:

- **`MOVE_TO` / `LINE_TO`**: endpoint `(x, y)`
- **`QUAD_TO`**: one control point `(cx0, cy0)` and endpoint `(x, y)`
- **`CURVE_TO`**: two control points `(cx0, cy0)`, `(cx1, cy1)`, and endpoint
  `(x, y)`
- **`CLOSE` / `END` / `PAD`**: no active coordinates

::: info
Coordinates are in em units: font design units divided by the font's
`unitsPerEm`.
:::

Quadratic curves are emitted as `QUAD_TO` without conversion to cubic. Every
element uses a six-value row to keep the tensor shape fixed; values in inactive
positions have no semantic meaning.

## Font face and character labels

### `font_idx`

`font_idx` is the font face class ID. Use it to look up the persistent face reference:

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
)

sample = dataset[0]

print(dataset.font_classes[sample.font_idx])
```

You will see output like:

```
FontRef(path='.../Aclonica-Regular.ttf', ttc_index=0)
```

### `character_idx`

`character_idx` is the character class ID. `character_classes` returns the
corresponding character. Run the following code to check the value:

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
)

sample = dataset[0]

print(dataset.character_classes[sample.character_idx])
```

You will see output like:

```
A
```
