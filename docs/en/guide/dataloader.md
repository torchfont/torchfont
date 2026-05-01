# DataLoader Integration

TorchFont datasets subclass `torch.utils.data.Dataset`, so they fit standard
PyTorch training loops.

## Quick sanity check (`batch_size=1`)

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="~/fonts")
sample = dataset[0]
print(sample.types.shape, sample.coords.shape)  # (seq_len,), (seq_len, 6)
print(sample.style_idx, sample.content_idx)
```

Use this only to check end-to-end wiring. For batching, use `collate_fn`.

## Recommended `collate_fn` for training

```python
import sys

from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn


dataset = GlyphDataset(root="~/fonts")
num_workers = 8
mp_context = "fork" if sys.platform.startswith("linux") else "spawn"

loader_kwargs = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": num_workers,
    "collate_fn": collate_fn,
}

if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
    loader_kwargs["multiprocessing_context"] = mp_context

loader = DataLoader(dataset, **loader_kwargs)
batch = next(iter(loader))

print(batch.types.shape)
print(batch.coords.shape)
print(batch.bitmap.shape)
print(batch.targets.shape)
print(batch.metrics.shape)
```

`num_workers > 0` enables worker prefetching and multiprocessing context.
Keep those options unset when `num_workers=0`.

| Platform | Recommended `multiprocessing_context` |
| -------- | ------------------------------------- |
| Linux    | `"fork"`                              |
| macOS    | `"spawn"` or `"forkserver"`           |
| Windows  | `"spawn"`                             |

## Using `Patchify`

`Patchify` splits each sample into fixed-size patches, which simplifies batching
logic.

```python
from torchfont.transforms import Compose, LimitSequenceLength, Patchify

transform = Compose([
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])
```

After this transform, `types.shape` becomes `(num_patches, 32)`. If you still
batch whole samples, `num_patches` can vary across samples, so `collate_fn`
may still need to pad at the sample level.
