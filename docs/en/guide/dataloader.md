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

Use this only to check end-to-end wiring. For batching, provide a small
`collate_fn` that pads the variable-length outline tensors.

## Recommended `collate_fn` for training

```python
import sys

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample


def transform(sample: GlyphSample):
    return sample.types, sample.coords


def collate_fn(batch):
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(root="~/fonts", transform=transform)
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

This `collate_fn` pads only the leading sequence dimension. If your
dataset transform returns extra trailing dimensions, those dimensions are
preserved while batching.
