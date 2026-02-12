# Google Fonts

`GoogleFonts` is a preset dataset for shallow-fetching the Google Fonts
repository and indexing it as TorchFont samples.

## Minimal example

```python
from torchfont.datasets import GoogleFonts

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    download=True,
    depth=1,
)

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")
print(f"commit={dataset.commit_hash}")
```

## Practical workflow

### 1. First sync (network)

```python
_ = GoogleFonts(root="data/google/fonts", ref="main", download=True)
```

### 2. Reuse local cache

```python
reused = GoogleFonts(root="data/google/fonts", ref="main", download=False)
print(reused.commit_hash)
```

## How `download` works

| Mode             | Network fetch | `ref` resolution source |
| ---------------- | ------------- | ----------------------- |
| `download=True`  | yes           | remote fetch result     |
| `download=False` | no            | local Git objects only  |

`download` controls only whether fetch is performed. In both modes, TorchFont
force-checks out `ref` into `root`.

If `root/.git` does not exist yet, `download=False` raises `FileNotFoundError`.
For each new cache directory, run once with `download=True`.

`depth` controls fetch history (`1` shallow by default, `0` for full history).
With `download=True`, `ref` must be a concrete branch ref (`main` or
`refs/heads/main`) or explicit `refs/...`.
Remote-tracking refs (`origin/main`) and ref expressions (`main~1`) are rejected.

## Use a dedicated `root` for Google Fonts

`GoogleFonts` is a preset of `FontRepo` with the Google Fonts URL.

If `root/.git` already exists, TorchFont reuses that existing repository and
requires the existing `origin` URL to match the preset URL. Use a dedicated
cache directory for Google Fonts (for example `data/google/fonts`) and do not
share it with other sources.
If the existing repository has no `origin` remote, initialization fails with
`ValueError`.

::: warning
Treat `root` as a cache directory. Local edits under `root` can be overwritten
by dataset initialization.
:::

## Narrow the indexed dataset

### Default `patterns`

When `patterns=None`, TorchFont uses:

```python
(
    "apache/*/*.ttf",
    "ofl/*/*.ttf",
    "ufl/*/*.ttf",
    "!ofl/adobeblank/AdobeBlank-Regular.ttf",
)
```

Pass `patterns` explicitly when you want to index only part of the repository.

### Limit character coverage

```python
dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    codepoint_filter=range(0x30, 0x3A),  # 0-9
    download=True,
    depth=1,
)
```

`codepoint_filter` removes unwanted characters during indexing.

## Reproducibility tip

Branches (for example `main`) move over time. For strict reproducibility, pin
`ref` to a commit hash or record `dataset.commit_hash` and reuse it.

```python
dataset = GoogleFonts(root="data/google/fonts", ref="main", download=True)
print(dataset.commit_hash)

# Later: pin exactly the same snapshot
repro = GoogleFonts(
    root="data/google/fonts",
    ref=dataset.commit_hash,
    download=False,
)
```

## Training pipeline example

```python
from collections.abc import Sequence
import sys

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GoogleFonts
from torchfont.transforms import Compose, LimitSequenceLength, Patchify

transform = Compose([
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    transform=transform,
    download=True,
)


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


num_workers = 8
loader_kwargs = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": num_workers,
    "collate_fn": collate_fn,
}

if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
    loader_kwargs["multiprocessing_context"] = (
        "fork" if sys.platform.startswith("linux") else "spawn"
    )

loader = DataLoader(dataset, **loader_kwargs)
```

::: tip Platform-specific multiprocessing

- Linux: `"fork"` is often fastest
- macOS: use `"spawn"` or `"forkserver"`
- Windows: use `"spawn"`
- If you set `num_workers=0`, remove `prefetch_factor` and
  `multiprocessing_context`

:::

## Related pages

- [Git Repositories](/en/guide/git-repos)
- [DataLoader Integration](/en/guide/dataloader)
- [Subset Extraction](/en/guide/subset)
