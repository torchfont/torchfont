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

Use this only to check end-to-end wiring. For batching, use `collate_outline`.

## Recommended `collate_outline` for training

```python
import sys

from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.utils import collate_outline


def transform(sample: GlyphSample):
    return sample.types, sample.coords


dataset = GlyphDataset(root="~/fonts", transform=transform)
num_workers = 8
mp_context = "fork" if sys.platform.startswith("linux") else "spawn"

loader_kwargs = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": num_workers,
    "collate_fn": collate_outline,
}

if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
    loader_kwargs["multiprocessing_context"] = mp_context

loader = DataLoader(dataset, **loader_kwargs)
types_t, coords_t = next(iter(loader))

print(types_t.shape)   # (64, L)
print(coords_t.shape)  # (64, L, 6)
```

`num_workers > 0` enables worker prefetching and multiprocessing context.
Keep those options unset when `num_workers=0`.

| Platform | Recommended `multiprocessing_context` |
| -------- | ------------------------------------- |
| Linux    | `"fork"`                              |
| macOS    | `"spawn"` or `"forkserver"`           |
| Windows  | `"spawn"`                             |

## Custom Sample Shapes

`collate_outline` pads only the leading sequence dimension. If your dataset transform
returns extra trailing dimensions, those dimensions are preserved while batching.
