# Using Git Repository Checkouts

TorchFont does not need to synchronize Git repositories itself in the main
workflow. Clone or update the repository outside TorchFont, then point
`GlyphDataset` at the checked-out directory.

## Basic form

```bash
git clone --depth 1 https://github.com/FortAwesome/Font-Awesome \
  data/fortawesome/font-awesome
```

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/fortawesome/font-awesome",
    patterns=("otfs/*.otf",),
)
```

## When to use this pattern

- you want to use existing OSS font assets directly
- you want reproducibility by recording the checkout commit yourself
- you want every corpus to enter TorchFont through the same local-directory API

## Pattern syntax

`patterns` follows gitignore-compatible matching.

| Pattern      | Meaning                                                     |
| ------------ | ----------------------------------------------------------- |
| `*.ttf`      | `.ttf` files matched by basename (including subdirectories) |
| `**/*.ttf`   | explicit recursive `.ttf` match from any depth              |
| `otfs/*.otf` | `.otf` files under `otfs/`                                  |
| `!*Bold*`    | exclude paths containing `Bold`                             |

::: info
`patterns` filters candidate paths first. TorchFont then keeps only files with
`.ttf`, `.otf`, `.ttc`, or `.otc` extensions.
:::

## Real examples

### Font Awesome

```python
GlyphDataset(
    root="data/fortawesome/font-awesome",
    patterns=("otfs/*.otf",),
)
```

### Material Design Icons

```python
GlyphDataset(
    root="data/google/material_design_icons",
    patterns=("variablefont/*.ttf",),
)
```

### Source Han Sans (with TTC)

```python
GlyphDataset(
    root="data/adobe-fonts/source-han-sans",
    patterns=("*.ttf.ttc",),
)
```

::: info
The `*.ttf.ttc` pattern is intentional. This repository contains TTC files with
names like `Something.ttf.ttc`.
:::

## Refresh workflow

Update the checkout with Git, then recreate the dataset instance:

```bash
git -C data/fortawesome/font-awesome fetch --depth 1 origin
git -C data/fortawesome/font-awesome checkout 7.x
```

TorchFont caches glyph metadata inside the native backend for the lifetime of a
dataset object, so create a new dataset after files on disk change.

## Reproducibility tip

Record the checkout commit outside TorchFont:

```bash
git -C data/fortawesome/font-awesome rev-parse HEAD
```

## DataLoader integration

Use the same `collate_fn` pattern described in
[DataLoader Integration](/en/guide/dataloader).
