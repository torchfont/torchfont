# トランスフォーム Utility

`torchfont.transforms` は glyph tensor を調整するための小さな utility 関数を提供します。
dataset item の整形は利用側の前処理コードで行います。

## quad_to_cubic

```python
from torchfont.transforms import quad_to_cubic
```

```python
types, coords = quad_to_cubic(types, coords)
```

`CommandType.QUAD_TO` を `CommandType.CURVE_TO` へ変換します。

- コマンド形状は変わりません
- 座標形状は変わりません（`(..., 6)`）
- 各 2 次セグメントの `[cx0, cy0, 0, 0, x, y]` は、直前終点を使って
  3 次制御点に書き換えられます
- `types` の最後の次元をシーケンス次元として扱い、先行次元は独立した
  シーケンスとして扱います

1 つの連続した outline を patch に分割し、patch 境界をまたいだ終点の連続性が
必要な場合は、分割前に `quad_to_cubic` を呼び出してください。

### 入出力

- 入力: `types=(...)`, `coords=(..., 6)`
- 出力: `types=(...)`, `coords=(..., 6)`
