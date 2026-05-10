# ユーティリティ API

`torchfont.utils` は可変長 glyph テンソルを batch 化する helper を提供します。

## `collate_outline`

```python
from torchfont.utils import collate_outline
```

```python
collate_outline(batch: Sequence[tuple[Tensor, Tensor]]) -> tuple[Tensor, Tensor]
```

dataset transform が返す `(types, coords)` ペアのリストを受け取り、
先頭シーケンス次元を batch 内最長に揃えてパディングします。
`torch.utils.data.DataLoader` の `collate_fn` 引数としてそのまま渡せます。

- `batch` は dataset transform が返す `(types, coords)` ペアのシーケンスです
- `batch` は非空である必要があり、空入力では `ValueError` を送出します
- padding は先頭シーケンス次元のみで、末尾次元はそのまま保持されます

### 入出力形状

- 入力: `(types, coords)` ペアのシーケンス（`types=(L, ...)`, `coords=(L, ...)`）
- 出力: `(types, coords)`（`types=(B, L, ...)`, `coords=(B, L, ...)`）

### 例

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
