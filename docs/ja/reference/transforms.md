# トランスフォーム API

`torchfont.transforms` は `(types, coords) -> (types, coords)` の形で前処理を組み立てるためのモジュールです。

## Compose

```python
from torchfont.transforms import Compose
```

```python
Compose(
    transforms: Sequence[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]],
)
```

複数トランスフォームを順番に適用します。

### 例（`Compose`）

```python
from torchfont.transforms import Compose, LimitSequenceLength, Patchify, QuadToCubic

transform = Compose([
    QuadToCubic(),
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])
```

---

## LimitSequenceLength

```python
from torchfont.transforms import LimitSequenceLength
```

```python
LimitSequenceLength(max_len: int)
```

`types[:max_len]` と `coords[:max_len]` を返します。

- パディングは行いません
- `max_len` を超えた後半は切り捨て

### 入出力（`LimitSequenceLength`）

- 入力: `types=(seq_len,)`, `coords=(seq_len, d)`
- 出力: `types=(min(seq_len, max_len),)`, `coords=(min(seq_len, max_len), d)`

---

## QuadToCubic

```python
from torchfont.transforms import QuadToCubic
```

```python
QuadToCubic()
```

`CommandType.QUAD_TO` を `CommandType.CURVE_TO` へ変換します。

- コマンド形状は変わりません
- 座標形状は変わりません（`(..., 6)`）
- 各 2 次セグメントの `[cx0, cy0, 0, 0, x, y]` は、直前終点を使って
  3 次制御点に書き換えられます

### 入出力（`QuadToCubic`）

- 入力: `types=(...)`, `coords=(..., 6)`
- 出力: `types=(...)`, `coords=(..., 6)`

---

## Patchify

```python
from torchfont.transforms import Patchify
```

```python
Patchify(patch_size: int)
```

シーケンス長が `patch_size` の倍数になるよう末尾をゼロ埋めし、パッチに並べ替えます。

### 入出力（`Patchify`）

- 入力: `types=(seq_len,)`, `coords=(seq_len, d)`
- 出力: `types=(num_patches, patch_size)`, `coords=(num_patches, patch_size, d)`
- `num_patches = ceil(seq_len / patch_size)`

### 備考

- `types` の埋め値は `0`（`pad`）
- `coords` の埋め値は `0.0`
- `patch_size` は `> 0` が必須（`0` や負値は実行時エラー）

### 例（`Patchify`）

```python
patchify = Patchify(patch_size=32)
patch_types, patch_coords = patchify(types, coords)
```
