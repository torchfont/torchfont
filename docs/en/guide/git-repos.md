# Using Git Repository Checkouts

TorchFont does not need to synchronize Git repositories itself in the main
workflow. Clone or update the repository outside TorchFont, then point
`GlyphDataset` at the checked-out directory.

## Basic form

```bash
git submodule update --init --depth 1 -- data/fortawesome/font-awesome
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
    patterns=("font/*.ttf", "font/*.otf"),
)
```

### Source Han Code JP (TTC)

```python
GlyphDataset(
    root="data/adobe/source-han-code-jp",
    patterns=("OTC/*.ttc",),
)
```

## Refresh workflow

Update the submodule checkout with Git, then recreate the dataset instance:

```bash
git submodule update --remote --depth 1 -- data/fortawesome/font-awesome
```

TorchFont keeps native indexing state for the lifetime of a dataset object.
Create a new dataset after files on disk change so that state stays in sync
with the checkout. If files change while a dataset instance is still in use,
results are undefined and may include incorrect samples or runtime errors.

## Reproducibility tip

Record the checkout commit outside TorchFont:

```bash
git -C data/fortawesome/font-awesome rev-parse HEAD
```

## DataLoader integration

Use the same local `collate_fn` pattern described in
[DataLoader Integration](/en/guide/dataloader).
