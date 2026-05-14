# Google Fonts

Use a local checkout of the Google Fonts repository, then open it with
`GlyphDataset`.

## Minimal example

```bash
git submodule update --init --depth 1 -- data/google/fonts
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

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")
```

## Why use this flow

- repository sync stays in normal Git tooling
- Google Fonts uses the same `GlyphDataset(root=...)` boundary as any other
  local corpus
- reproducibility comes from the checkout commit you record externally

## Recommended `patterns`

These include/exclude rules are a good default when using `GlyphDataset`
directly:

```python
(
    "apache/*/*.ttf",
    "ofl/*/*.ttf",
    "ufl/*/*.ttf",
    "!ofl/adobeblank/AdobeBlank-Regular.ttf",
)
```

Pass a narrower pattern when you only want part of the corpus.

### Limit character coverage

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=("ofl/*/*.ttf",),
    codepoints=range(0x30, 0x3A),  # 0-9
)
```

`codepoints` removes unwanted characters during indexing.

## Update workflow

Update the submodule checkout with Git, then recreate the dataset:

```bash
git submodule update --remote --depth 1 -- data/google/fonts
git -C data/google/fonts rev-parse HEAD
```

Record the commit hash externally when you need exact reproducibility.

## Training pipeline example

```python
import sys

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import patchify, quad_to_cubic, remove_overlaps, render_bitmap


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor, Tensor]:
    types = sample.types
    coords = sample.coords
    types, coords = remove_overlaps(types, coords)
    types, coords = types[:512], coords[:512]
    types, coords = quad_to_cubic(types, coords)
    bitmap = render_bitmap(types, coords)
    patch_types, patch_coords = patchify(types, coords, patch_size=32)
    return patch_types, patch_coords, bitmap


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    types = pad_sequence([types for types, _, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords, _ in batch], batch_first=True)
    bitmaps = torch.stack([bitmap for _, _, bitmap in batch])
    return types, coords, bitmaps


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=("ofl/*/*.ttf",),
    transform=transform,
)

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
