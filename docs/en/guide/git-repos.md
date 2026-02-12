# Fetching Fonts from Git Repositories

Use `FontRepo` when you want to synchronize font files from an arbitrary Git
repository and index them as a dataset.

## Basic form

```python
from torchfont.datasets import FontRepo

dataset = FontRepo(
    root="data/fortawesome/font-awesome",
    url="https://github.com/FortAwesome/Font-Awesome",
    ref="7.x",
    patterns=("otfs/*.otf",),
    download=True,
    depth=1,
)

print(dataset.commit_hash)
print(len(dataset), len(dataset.style_classes), len(dataset.content_classes))
```

## When to use `FontRepo`

- you want to use existing OSS font assets directly
- you want reproducibility by pinning a commit hash
- you need non-Google-Fonts sources in a unified pipeline

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
FontRepo(
    root="data/fortawesome/font-awesome",
    url="https://github.com/FortAwesome/Font-Awesome",
    ref="7.x",
    patterns=("otfs/*.otf",),
    download=True,
    depth=1,
)
```

### Material Design Icons

```python
FontRepo(
    root="data/google/material_design_icons",
    url="https://github.com/google/material-design-icons",
    ref="master",
    patterns=("variablefont/*.ttf",),
    download=True,
    depth=1,
)
```

### Source Han Sans (with TTC)

```python
FontRepo(
    root="data/adobe-fonts/source-han-sans",
    url="https://github.com/adobe-fonts/source-han-sans",
    ref="release",
    patterns=("*.ttf.ttc",),
    download=True,
    depth=1,
)
```

::: info
The `*.ttf.ttc` pattern is intentional. This repository contains TTC files with
names like `Something.ttf.ttc`.
:::

## `download` behavior

- `download=True`: fetch from remote, then force-checkout `ref`
- `download=False`: skip fetch, resolve `ref` locally, then force-checkout it
  (requires a locally resolvable `ref`)
- With `download=True`, `ref` must be a concrete branch ref
  (`main` or `refs/heads/main`) or explicit `refs/...`.
  Remote-tracking refs (`origin/main`) and ref expressions (`main~1`) are rejected.

If `root/.git` does not exist yet, `download=False` raises `FileNotFoundError`.
For a new cache directory, run once with `download=True`.

## Fetch depth (`depth`)

- `depth=1` (default): shallow fetch
- `depth=0`: fetch full history

Use `depth=0` when you need full history available locally (for example, when
you plan to resolve ancestry expressions later with `download=False`).
Keep `depth=1` for faster sync in normal dataset use.

## Important: `root` must match `url`

If `root/.git` already exists, TorchFont reuses that repository. The existing
`origin` URL must match the `url` argument; otherwise initialization fails with
`ValueError`.
If the existing repository has no `origin` remote, initialization also fails
with `ValueError`.

::: warning
`root` is checked out with a force strategy in both modes. Keep `root` as a
cache directory; local edits under `root` may be overwritten.
:::

::: tip Best practices

- For reproducible experiments, record `commit_hash` and rerun with
  `ref=<that_hash>`.
- If this is your first run for a given `root`, start with `download=True`.
- Use one cache directory (`root`) per source repository.

:::

## DataLoader integration

`FontRepo` extends `FontFolder`, so you can use the same `collate_fn` pattern
from [DataLoader Integration](/en/guide/dataloader).
