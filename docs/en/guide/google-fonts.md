# Google Fonts

Use a local checkout of the Google Fonts repository, then open it with
`GlyphDataset`.

## Minimal example

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

Update the checkout with Git, then recreate the dataset:

```bash
git -C data/google/fonts pull --ff-only
git -C data/google/fonts rev-parse HEAD
```

Record the commit hash externally when you need exact reproducibility.

## Training pipeline example

```python
import sys

from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.transforms import Compose, LimitSequenceLength, Patchify
from torchfont.utils import collate_fn

transform = Compose([
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])

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
