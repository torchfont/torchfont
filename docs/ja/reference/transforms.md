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

## patchify

```python
from torchfont.transforms import patchify
```

```python
patch_types, patch_coords = patchify(types, coords, patch_size=32)
```

1-D のグリフシーケンスを `patch_size` の倍数にゼロパディングし、
連続したパッチに分割します。

- `patch_size` は 1 以上が必要です
- `seq_len % patch_size != 0` の場合のみ末尾にゼロが追加されます
- patch 境界をまたいだ終点の連続性が必要な場合は、`patchify` の前に
  `quad_to_cubic` を呼び出してください

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(num_patches, patch_size)`, `coords=(num_patches, patch_size, 6)`

## render_bitmap

```python
from torchfont.transforms import render_bitmap
```

```python
bitmap = render_bitmap(types, coords, size=64)
```

グリフアウトラインをグレースケールビットマップテンソルへレンダリングします。
グリフは自動スケーリングされ、各辺 4 ピクセルのパディングを持つキャンバスに中央配置されます。

- `size` は 1〜4096 の整数（デフォルト: 64）
- patchify 前のクリップ済みアウトラインを渡すと元の形状を正確に再現できます

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `uint8` テンソル、形状 `(size, size)`、値域 `[0, 255]`
