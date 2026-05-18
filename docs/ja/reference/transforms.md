# トランスフォーム Utility

`torchfont.transforms` は glyph tensor を調整するための小さな utility 関数を提供します。
dataset item の整形は利用側の前処理コードで行います。

## quad_to_cubic

```python
from torchfont.transforms import quad_to_cubic
```

```python
types, coords = quad_to_cubic(types, coords)
# 1 つの連続した outline シーケンスでは:
types, coords = quad_to_cubic(types, coords, merge_curves=True)
```

`ElementType.QUAD_TO` を `ElementType.CURVE_TO` へ変換します。

- element type の形状は変わりません
- 座標形状は変わりません（`(..., 6)`）
- 各 2 次セグメントの `[cx0, cy0, 0, 0, x, y]` は、直前終点を使って
  3 次制御点に書き換えられます
- `types` の最後の次元をシーケンス次元として扱い、先行次元は独立した
  シーケンスとして扱います

1 つの連続した outline を patch に分割し、patch 境界をまたいだ終点の連続性が
必要な場合は、分割前に `quad_to_cubic` を呼び出してください。

`merge_curves=True` を指定すると、変換直後に復元可能な隣接カーブと共線の line を
同じ Rust 呼び出し内でまとめます。`cubic_to_quad` の後段で特に有用です。
マージ後は outline が短くなることがあるため、このモードは batched 入力ではなく
1 つの連続した outline シーケンスを受け取ります。

### 入出力

- 入力: `types=(...)`, `coords=(..., 6)`
- 出力: `types=(...)`, `coords=(..., 6)`
- `merge_curves=True` の場合: 入力 `types=(N,)`, `coords=(N, 6)`、出力
  `types=(M,)`, `coords=(M, 6)`

## cubic_to_quad

```python
from torchfont.transforms import cubic_to_quad
```

```python
types, coords = cubic_to_quad(types, coords)
```

fonttools cu2qu と同じ近似方針で、`ElementType.CURVE_TO` を 2 次 spline へ変換します。

- 1 つの連続した outline シーケンスを受け取ります
- 1 つの cubic が複数の `ElementType.QUAD_TO` に展開されることがあります
- 隣接する 2 次制御点の中点が暗黙の on-curve 点になります

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(M,)`, `coords=(M, 6)`

## merge_curves

```python
from torchfont.transforms import merge_curves
```

```python
types, coords = merge_curves(types, coords)
```

隣接セグメントを 1 つの親形状として復元できる場合にまとめます。

- 分割された cubic は、復元誤差が許容範囲内なら 1 つの cubic に戻します
- 分割された quadratic は、復元誤差が許容範囲内なら 1 つの quadratic に戻します
- 同方向へ進む連続した共線の `LineTo` は 1 つにまとめます
- subpath 境界は保持します

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(M,)`, `coords=(M, 6)`


## remove_overlaps

```python
from torchfont.transforms import remove_overlaps
```

```python
types, coords = remove_overlaps(types, coords)
```

Skia PathOps を使い、winding に基づく hole を保ったまま重なったグリフ subpath を統合します。

- 1 つの連続した outline シーケンスを受け取ります
- 重なり内部の edge を除去し、新しい可変長 outline を返します
- Skia PathOps が簡約できない outline は元のまま返します

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(M,)`, `coords=(M, 6)`

## normalize_subpath_start_points

```python
from torchfont.transforms import normalize_subpath_start_points
```

```python
types, coords = normalize_subpath_start_points(types, coords)
```

各 subpath の開始点を、辞書順で最小の `(x, y)` 終点へ移します。

- closed subpath のみを変更し、open subpath は変更しません
- 回転が元の close edge をまたぐ場合、その暗黙 edge を `LineTo` として実体化します
- 表す形状は保持します。元の開始点以外へ回す場合は `LineTo` が 1 つ増えることがあります

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(M,)`, `coords=(M, 6)`

## randomize_subpath_start_points

```python
from torchfont.transforms import randomize_subpath_start_points
```

```python
types, coords = randomize_subpath_start_points(types, coords)
```

各 subpath の開始終点を一様ランダムに選びます。

- subpath の開始点にモデルを依存させたくない場合の augmentation に使えます
- closed subpath のみを変更し、open subpath は変更しません
- `generator`: 再現性のためのオプション `torch.Generator`

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(M,)`, `coords=(M, 6)`

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

## horizontal_flip

```python
from torchfont.transforms import horizontal_flip
```

```python
types, coords = horizontal_flip(types, coords)
```

グリフアウトラインを tight bounding-box の中心を軸に水平反転します。

- on-curve 終点と off-curve 制御点の両方を変換します
- 座標が 0 の element type（CLOSE、END、PAD）は変更しません
- 閉じた subpath の巻き順はデフォルトで保持します
- 反転後の巻き順をそのまま使うには `preserve_winding=False` を指定します
- 開いた subpath は反転しますが、走査方向は反転しません

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)`, `coords=(N, 6)`

## vertical_flip

```python
from torchfont.transforms import vertical_flip
```

```python
types, coords = vertical_flip(types, coords)
```

グリフアウトラインを tight bounding-box の中心を軸に垂直反転します。

- on-curve 終点と off-curve 制御点の両方を変換します
- 座標が 0 の element type（CLOSE、END、PAD）は変更しません
- 閉じた subpath の巻き順はデフォルトで保持します
- 反転後の巻き順をそのまま使うには `preserve_winding=False` を指定します
- 開いた subpath は反転しますが、走査方向は反転しません

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)`, `coords=(N, 6)`

## affine

```python
from torchfont.transforms import affine
```

```python
types, coords = affine(types, coords, angle=15.0, translate=(0.05, 0.0), scale=0.9, shear=5.0)
```

グリフアウトラインに決定論的なアフィン変換を適用します。

tight bounding-box の中心を基準に一様スケール・x-shear・回転を合成し、
`translate` を適用します。すべてのアクティブな制御点と終点を変換します。
座標が 0 の element type（CLOSE、END、PAD）は変更しません。

- `angle`: 反時計回りの回転角度（単位: 度、デフォルト: `0.0`）
- `translate`: UPM 正規化単位での平行移動 `(tx, ty)`（デフォルト: `(0.0, 0.0)`）
- `scale`: 一様スケール係数（正数必須、デフォルト: `1.0`）
- `shear`: x-shear 角度（単位: 度、デフォルト: `0.0`）

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)` (変更なし), `coords=(N, 6)`

## random_horizontal_flip

```python
from torchfont.transforms import random_horizontal_flip
```

```python
types, coords = random_horizontal_flip(types, coords, p=0.5)
```

確率 `p` で `horizontal_flip` をランダムに適用します。

- `p`: 反転確率（デフォルト: `0.5`）
- `preserve_winding`: 反転後も閉じた subpath の巻き順を保持します（デフォルト: `True`）
- `generator`: 再現性のためのオプション `torch.Generator`

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)`, `coords=(N, 6)`

## random_vertical_flip

```python
from torchfont.transforms import random_vertical_flip
```

```python
types, coords = random_vertical_flip(types, coords, p=0.5)
```

確率 `p` で `vertical_flip` をランダムに適用します。

- `p`: 反転確率（デフォルト: `0.5`）
- `preserve_winding`: 反転後も閉じた subpath の巻き順を保持します（デフォルト: `True`）
- `generator`: 再現性のためのオプション `torch.Generator`

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)`, `coords=(N, 6)`

## random_affine

```python
from torchfont.transforms import random_affine
```

```python
types, coords = random_affine(
    types, coords,
    degrees=15.0,
    translate=(0.05, 0.05),
    scale=(0.9, 1.1),
    shear=5.0,
)
```

指定した範囲から一様サンプリングしたランダムなアフィン変換を適用します。

- `degrees`: 回転範囲（度）。単一の float `d` を指定すると `[-d, d]` になります
- `translate`: UPM 正規化単位での最大絶対平行移動 `(max_dx, max_dy)`。
  各軸を `[-max_d, max_d]` からサンプリングします（デフォルト: 平行移動なし）
- `scale`: スケール範囲 `(min, max)`。両値は正数必須（デフォルト: スケールなし）
- `shear`: x-shear 範囲（度）。`degrees` と同じ書式（デフォルト: `0.0`）
- `generator`: 再現性のためのオプション `torch.Generator`

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)` (変更なし), `coords=(N, 6)`

## random_coord_jitter

```python
from torchfont.transforms import random_coord_jitter
```

```python
types, coords = random_coord_jitter(types, coords, std=0.005)
```

各アクティブな outline 座標に独立したガウスノイズを加算します。

- `std`: UPM 正規化単位での標準偏差。`0.005` は 1000-UPM フォントで
  約 5 フォントユニットに相当します
- 座標が 0 の element type（CLOSE、END、PAD）と未使用の座標列は変更しません
- `generator`: 再現性のためのオプション `torch.Generator`

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `types=(N,)` (変更なし), `coords=(N, 6)`

## render_bitmap

```python
from torchfont.transforms import render_bitmap
```

```python
bitmap = render_bitmap(types, coords, size=64, mode="bbox_square")
```

グリフアウトラインをグレースケールビットマップテンソルへレンダリングします。
`mode` に応じた座標変換で出力ビットマップへ配置します。

- `size` は 1〜4096 の整数（デフォルト: 64）
- `mode="fixed"` は UPM 正規化済みの固定範囲 `[-0.25, 1.25] x [-0.25, 1.25]` に配置
- `mode="bbox"` は fixed と同じ座標スケールを保ち、tight bbox に合わせた可変サイズのビットマップを返します
- `mode="bbox_square"` は tight bbox を縦横比を保って正方形内に中央配置（デフォルト）
- patchify 前のクリップ済みアウトラインを渡すと元の形状を正確に再現できます

### 入出力

- 入力: `types=(N,)`, `coords=(N, 6)`
- 出力: `uint8` テンソル、値域 `[0, 255]`。`fixed` / `bbox_square` は
  `(size, size)`、`bbox` は可変の `(height, width)`
