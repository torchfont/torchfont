# Quickstart

This page gets you from installation to a working dataset and DataLoader setup.

## Requirements

- Python 3.10+
- PyTorch 2.3+

## 1. Install

::: code-group

```bash [uv (Recommended)]
uv add torchfont
```

```bash [pip]
pip install torchfont
```

:::

## 2. Load local fonts

`GlyphDataset` scans local directories for `.ttf`, `.otf`, `.ttc`, and `.otc`
files.

```python
from torchfont.datasets import GlyphDataset

# root must point to an existing directory
# e.g. root="~/fonts" (or "tests/fonts" if you cloned this repository)
dataset = GlyphDataset(
    root="~/fonts",
    patterns=("*.ttf",),
    codepoints=range(0x20, 0x7F),
)

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")

sample = dataset[0]
print(sample.types.shape)   # (seq_len,)
print(sample.coords.shape)  # (seq_len, 6)
```

::: warning
`GlyphDataset` resolves `root` to an absolute path during initialization. If the
path does not exist, initialization fails.
:::

## 3. Plug into DataLoader

Glyph sequences are variable-length, so the built-in
`torchfont.utils.collate_fn` is the standard starting point.

```python
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn


dataset = GlyphDataset(
    root="~/fonts",
    patterns=("*.ttf",),
    codepoints=range(0x20, 0x7F),
)
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
print(batch.types.shape)
print(batch.coords.shape)
print(batch.mask.shape)
```

## 4. Scale up with a local checkout

Sync the repository outside TorchFont, then point `GlyphDataset` at the
checked-out directory.

```bash
git clone --depth 1 https://github.com/google/fonts data/google/fonts
```

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
)

print(len(dataset), len(dataset.style_classes), len(dataset.content_classes))
```

TorchFont treats any checkout as a normal local directory. Use Git or another
sync tool to refresh the checkout, then recreate the dataset instance so the
native indexing state stays in sync with the files. If files change while a
dataset instance is still in use, results are undefined and may include
incorrect samples or runtime errors.

## Common Next Tweaks

- Limit long sequences: `LimitSequenceLength(max_len=...)`
- Convert quadratic segments to cubic: `QuadToCubic()`
- Use fixed-size inputs: `Patchify(patch_size=...)`
- Narrow the dataset scope: `codepoints=` and `patterns=`

## Read Next

- [Glyph Data Format](/en/guide/glyph-data-format)
- [DataLoader Integration](/en/guide/dataloader)
- [Git Repository Checkouts](/en/guide/git-repos)
- [Google Fonts](/en/guide/google-fonts)
