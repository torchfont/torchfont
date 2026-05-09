# トランスフォーム Utility

`torchfont.transforms` は glyph tensor を調整するための小さな utility 関数を提供します。
dataset item の整形は利用側の前処理コードで行います。

## QuadToCubic

```python
from torchfont.transforms import QuadToCubic
```

```python
types, coords = QuadToCubic(types, coords)
```

`CommandType.QUAD_TO` を `CommandType.CURVE_TO` へ変換します。

- コマンド形状は変わりません
- 座標形状は変わりません（`(..., 6)`）
- 各 2 次セグメントの `[cx0, cy0, 0, 0, x, y]` は、直前終点を使って
  3 次制御点に書き換えられます

### 入出力

- 入力: `types=(...)`, `coords=(..., 6)`
- 出力: `types=(...)`, `coords=(..., 6)`
