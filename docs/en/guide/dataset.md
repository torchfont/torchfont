# Building a GlyphDataset

## Loading the dataset

Once Google Fonts is set up, load it with `GlyphDataset`. Run the following code:

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="data/google/fonts")

print(f"{len(dataset)=}")
print(f"{len(dataset.style_classes)=}")
print(f"{len(dataset.content_classes)=}")
```

The output will look like this. `style_classes` is the number of distinct font
styles, and `content_classes` is the number of unique codepoints in the dataset.

```
len(dataset)=13745683
len(dataset.style_classes)=9113
len(dataset.content_classes)=1112000
```

## Filtering with `patterns`

Use `patterns` to limit which font files are loaded. For Google Fonts, the
following patterns are a good default:

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
)
```

The reasoning behind each pattern:

- **`apache/*/*.ttf`, `ofl/*/*.ttf`, `ufl/*/*.ttf`**: These are the only three
  license directories in Google Fonts. The repository contains other directories
  that may include test fonts and similar files, so only these three are targeted
  explicitly.
- **`/*/*.ttf` (not `/**/*.ttf`)**: Google Fonts distributes fonts in TTF format
  only. Using `/**/*.ttf` would also match extra fonts placed in subdirectories,
  so only one level of depth is used.
- **`!ofl/adobeblank/AdobeBlank-Regular.ttf`**: AdobeBlank is designed for
  fallback rendering verification. It maps an enormous number of codepoints but
  contains no real glyph outlines. Including it in a training dataset severely
  degrades data quality, so it is advisable to exclude it.

Running this code produces the following output. Compared to loading without
patterns, both the style count and content classes (codepoints) are reduced:

```
len(dataset)=12460609
len(dataset.style_classes)=8951
len(dataset.content_classes)=114254
```

### Filtering by codepoint

Use `codepoints` to exclude unwanted codepoints during indexing.

To restrict to the 26 uppercase letters (A–Z), add a `codepoints` argument as follows:

```python
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
```

The output will look like this:

```
len(dataset)=220939
len(dataset.style_classes)=8498
len(dataset.content_classes)=26
```

To restrict to the Basic Multilingual Plane (BMP, U+0000–U+FFFF):

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x0000, 0x10000),
)
```

The output will look like this:

```
len(dataset)=12027967
len(dataset.style_classes)=8951
len(dataset.content_classes)=60004
```
