# ユーティリティ API

`torchfont.utils` は可変長 glyph sample を batch 化する helper を提供します。

## GlyphBatch

```python
from torchfont.utils import GlyphBatch
```

`collate_fn` の返り値となる構造化 batch 型です。

| 要素                 | 型                  | 形状        |
| -------------------- | ------------------- | ----------- |
| `batch.types`        | `torch.LongTensor`  | `(B, L, ...)` |
| `batch.coords`       | `torch.FloatTensor` | `(B, L, ...)` |
| `batch.style_idx`    | `torch.LongTensor`  | `(B,)`        |
| `batch.content_idx`  | `torch.LongTensor`  | `(B,)`        |
| `batch.mask`         | `torch.BoolTensor`  | `(B, L)`      |

`batch.mask` は有効なシーケンス位置で `True`、padding 部分で `False` です。
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
- `batch` 内の sample は `types.shape[1:]` と `coords.shape[1:]` を
  そろえる必要があり、互換性のない layout は `ValueError` を送出します。

### 例

```python
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn

dataset = GlyphDataset(root="~/fonts")
loader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

batch = next(iter(loader))
print(batch.types.shape)
print(batch.mask.shape)
```
