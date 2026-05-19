# Glyph Data Format

## Accessing a sample

Access a sample from the dataset created in the previous chapter. Run the following code:

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

print(sample.types)        # element type sequence
print(sample.coords)       # coordinates sequence
print(sample.style_idx)    # style class ID
print(sample.content_idx)  # content class ID
```

The return value is a `GlyphSample` dataclass. Access its fields by name.

## Outline model

A glyph outline is represented as a sequence of path elements.

- **Path element**: the smallest unit — one element type plus one coordinates row
- **Subpath**: a sequence of path elements representing one continuous curve that makes up a glyph
- **Outline**: a sequence of path elements representing the contour of one glyph

`sample.types` is a `(seq_len,)` `LongTensor` of element types as integers. `sample.coords` is a `(seq_len, 6)` `FloatTensor` of coordinates as floats.

## Element type

Element types are defined in `ElementType`. Run the following code to see the mapping between values and names:

```python
from torchfont.datasets import GlyphDataset
from torchfont.io import ElementType

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

print(sample.types)
print(ElementType(sample.types[0].item()).name)
```

You will see output like:

```
tensor([1, 2, 3, ..., 5, 6])
MOVE_TO
```

The seven types are `MoveTo`, `LineTo`, `QuadTo`, `CurveTo`, `Close`, `End`, and `Pad`.

- `ElementType.END` marks the end of the sequence
- `ElementType.PAD` is mainly introduced by `pad_sequence` or custom padding

## Coordinates

Each path element uses a 6D coordinates vector. Run the following code to inspect the shape and contents:

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

print(sample.coords.shape)
print(sample.coords[0])
```

You will see output like:

```
torch.Size([seq_len, 6])
tensor([cx0, cy0, cx1, cy1, x, y])
```

Which dimensions are used depends on the element type:

- **`MoveTo` / `LineTo`**: endpoint `(x, y)` only; control points are zero
- **`QuadTo`**: one control point `(cx0, cy0)` and endpoint `(x, y)`; `cx1`, `cy1` are zero
- **`CurveTo`**: two control points `(cx0, cy0)`, `(cx1, cy1)`, and endpoint `(x, y)`
- **`Close` / `End` / `Pad`**: all zeros

::: info
Coordinates are normalized by the font `units_per_em`.
:::

Quadratic curves are emitted as `QuadTo` without conversion to cubic. To keep tensor shape fixed, `QuadTo` uses `[cx0, cy0, 0, 0, x, y]`.

## Style and content labels

### `style_idx`

`style_idx` is the style class ID. Run the following code to look up the corresponding name:

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

print(dataset.style_classes[sample.style_idx])
```

You will see output like:

```
Aclonica Regular
```

### `content_idx`

`content_idx` is the content class ID. `content_classes` returns the corresponding character. Run the following code to check the value:

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

print(dataset.content_classes[sample.content_idx])
```

You will see output like:

```
A
```
