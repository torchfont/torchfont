# ユーティリティ API

`torchfont.utils` は可変長 glyph sample を batch 化する helper を提供します。

## GlyphBatch

```python
from torchfont.utils import GlyphBatch
```

`collate_fn` の返り値となる構造化 batch 型です。

| 要素              | 型                  | 形状          |
| ----------------- | ------------------- | ------------- |
| `batch.types`     | `torch.LongTensor`  | `(B, L, ...)` |
| `batch.coords`    | `torch.FloatTensor` | `(B, L, ...)` |
| `batch.targets`   | `torch.LongTensor`  | `(B, 2)`      |
| `batch.metrics`   | `torch.FloatTensor` | `(B, 15)`     |

`batch.targets[:, 0]` が style インデックス、`batch.targets[:, 1]` が content
インデックスです。`batch.metrics` の列順序は `GlyphSample.metrics` と同じです。
`collate_fn` が padding するのは先頭のシーケンス次元 `L` のみで、`Patchify`
などの前処理で増えた末尾次元はそのまま保持されます。

## `collate_fn`

```python
from torchfont.utils import collate_fn
```

```python
collate_fn(batch: Sequence[GlyphSample]) -> GlyphBatch
```

可変長 glyph sample の先頭シーケンス次元だけを batch 内最長に合わせて
padding し、`GlyphBatch` を返します。

- `batch` は非空である必要があり、空入力では `ValueError` を送出します。

### 例

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
