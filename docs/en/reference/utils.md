# Utility API

`torchfont.utils` provides batching helpers for variable-length glyph samples.

## GlyphBatch

```python
from torchfont.utils import GlyphBatch
```

Structured return type for `collate_fn`.

| Field           | Type                | Shape         |
| --------------- | ------------------- | ------------- |
| `batch.types`   | `torch.LongTensor`  | `(B, L, ...)` |
| `batch.coords`  | `torch.FloatTensor` | `(B, L, ...)` |
| `batch.bitmap`  | `torch.ByteTensor`  | `(B, 64, 64)` |
| `batch.targets` | `torch.LongTensor`  | `(B, 2)`      |
| `batch.metrics` | `torch.FloatTensor` | `(B, 15)`     |

`batch.targets[:, 0]` holds style indices and `batch.targets[:, 1]` holds content
indices. `batch.metrics` columns follow the same order as `GlyphSample.metrics`.
`collate_fn` only pads along the sequence dimension `L`; trailing dimensions
produced by transforms like `Patchify` are preserved.

## `collate_fn`

```python
from torchfont.utils import collate_fn
```

```python
collate_fn(batch: Sequence[GlyphSample]) -> GlyphBatch
```

Pads the leading variable-length sequence dimension to the longest sample in the
batch and returns a `GlyphBatch`.

- `batch` must be non-empty; empty input raises `ValueError`

### Example

```python
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn

dataset = GlyphDataset(root="~/fonts")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

batch = next(iter(loader))
print(batch.types.shape)
print(batch.metrics.shape)
```
