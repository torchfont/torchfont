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

`FontFolder` scans local directories for `.ttf`, `.otf`, `.ttc`, and `.otc`
files.

```python
from torchfont.datasets import FontFolder

# root must point to an existing directory
# e.g. root="~/fonts" (or "tests/fonts" if you cloned this repository)
dataset = FontFolder(root="~/fonts")

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")

types, coords, style_idx, content_idx = dataset[0]
print(types.shape)   # (seq_len,)
print(coords.shape)  # (seq_len, 6)
```

::: warning
`FontFolder` resolves `root` to an absolute path during initialization. If the
path does not exist, initialization fails.
:::

## 3. Plug into DataLoader

Glyph sequences are variable-length, so a custom `collate_fn` with padding is
the standard pattern.

```python
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import FontFolder


def collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    types_list = [t for t, _, _, _ in batch]
    coords_list = [c for _, c, _, _ in batch]
    style_list = [s for _, _, s, _ in batch]
    content_list = [c for _, _, _, c in batch]

    types = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    style_idx = torch.as_tensor(style_list, dtype=torch.long)
    content_idx = torch.as_tensor(content_list, dtype=torch.long)
    return types, coords, style_idx, content_idx


dataset = FontFolder(root="~/fonts")
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
print([x.shape for x in batch[:2]])
```

## 4. Try Google Fonts at scale

```python
from torchfont.datasets import GoogleFonts

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    download=True,
)

print(dataset.commit_hash)
print(len(dataset), len(dataset.style_classes), len(dataset.content_classes))
```

- `download=True`: fetch from remote, then force-checkout `ref`
- `download=False`: skip fetch, resolve `ref` locally, then force-checkout it
  (requires a locally resolvable `ref`)
- `depth=1`: shallow fetch (default), `depth=0`: full history
- With `download=True`, `ref` must be a concrete branch ref
  (`main` or `refs/heads/main`) or explicit `refs/...` path.
  Remote-tracking refs (`origin/main`) and ref expressions (`main~1`) are rejected.

If `root/.git` does not exist yet, `download=False` raises `FileNotFoundError`.
Run once with `download=True` for each new cache directory.

::: warning
`FontRepo`/`GoogleFonts` use a force checkout strategy to align `root` with
`ref`. Keep this directory as a dataset cache; local edits under `root` may be
overwritten.
:::

## Common Next Tweaks

- Limit long sequences: `LimitSequenceLength(max_len=...)`
- Use fixed-size inputs: `Patchify(patch_size=...)`
- Narrow the dataset scope: `codepoint_filter=` and `patterns=`

## Read Next

- [Glyph Data Format](/en/guide/glyph-data-format)
- [DataLoader Integration](/en/guide/dataloader)
- [Google Fonts](/en/guide/google-fonts)
