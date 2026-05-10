# Utility API

`torchfont.utils` provides batching helpers for variable-length glyph tensors.

## `collate_outline`

```python
from torchfont.utils import collate_outline
```

```python
collate_outline(batch: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]
```

Pads the leading variable-length sequence dimension to the longest sample in the
batch. Suitable for use as the `collate_fn` argument of
`torch.utils.data.DataLoader`.

- `batch` is a sequence of `(types, coords)` pairs returned by a dataset transform
- `batch` must be non-empty; empty input raises `ValueError`
- padding is zero-filled and added only to the leading sequence dimension;
  trailing dimensions are preserved

### I/O Shape

- input: sequence of `(types, coords)` where `types=(L, ...)` and `coords=(L, ...)`
- output: `(types, coords)` where `types=(B, L, ...)` and `coords=(B, L, ...)`

### Example

```python
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_outline

dataset = GlyphDataset(root="~/fonts", transform=lambda s: (s.types, s.coords))
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_outline)

types_t, coords_t = next(iter(loader))
print(types_t.shape)   # (32, L)
print(coords_t.shape)  # (32, L, 6)
```
