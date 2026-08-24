# Transform

TorchFont は Glyph の読み込み、変形、描画を組み合わせるための Transform を提供します。
結果を画像パイプラインへ渡す場合は、任意で TorchVision を組み合わせられます。

## データ型

```python
from torchfont import GlyphData, GlyphIdData, Outline
```

`Outline(types, coords)` は不可分な二つのテンソルを一つにまとめます。
`GlyphData[T]` は変換中の Payload、Glyph 参照、Variation Location、Target を保持します。
`GlyphIdData[T]` はグリフ ID の Sample に対応する型で、コードポイントの Target を除いた
同じ Field を保持します。どちらも他の Field を失わずに、Payload を `Outline` から
ビットマップテンソルへ変換できます。

## 読み込みと合成

```python
from torchfont.transforms import (
    Compose,
    LoadGlyph,
    RandomApply,
    RandomSplitSegments,
    RemoveOverlaps,
)

transform = Compose(
    [
        LoadGlyph(),
        RandomApply(RandomSplitSegments(split_probability=0.2), p=0.5),
        RemoveOverlaps(),
    ]
)

data = transform(sample)
outline = data.data
```

`LoadGlyph` は一つの `GlyphSample`、`GlyphIdSample`、`GlyphRef` を読み込みます。
`GlyphSample` は `GlyphData[Outline]` に、`GlyphIdSample` は `GlyphIdData[Outline]` に、
参照単体は `Outline` になります。
`LoadGlyph` は、`location="random"` を指定しない限り Face の Default Location を使います。
Random Policy はいずれの入力に対しても位置を 1 点抽出し、返り値の `location` に保存します。
Static Face では空の位置になります。
Dataset Sample に対しては、返り値の並列な `weight`、`width`、`italic`、
`slant`、`optical_size` Target も解決します。

Transform はネストした入力を受け取り、その構造を保ちます。一回の呼び出しに含まれる
対応する複数 Outline には、同じランダムパラメーターが適用されます。
独立したサンプルには Transform を個別に適用します。確率的 Transform は PyTorch の
デフォルト RNG を使うため、`torch.manual_seed` と `DataLoader` ワーカーのシードが通常どおり
機能します。
組み込み Transform はマルチプロセスの DataLoader でも使用できます。`Compose` には
`nn.Module` の Transform を渡します。独自 Transform も通常の Callable ではなく
`nn.Module` の Subclass として定義してください。空の `Compose` は入力を変更しません。
複数の Transform を `RandomApply` でまとめる場合は、内側に `Compose` を置きます。

`eval()` を呼んでもランダムな Data Augmentation は無効になりません。評価時には
決定論的な Pipeline を使用してください。

`RandomApply(transform, p)` は一つの Transform を適用するか制御します。
`RandomSplitSegments.split_probability` などは、適用された Transform 内部の挙動を
制御します。

## 組み込み Transform

| 分類 | Transform |
| --- | --- |
| 読み込み | `LoadGlyph` |
| コンテナ | `Compose`, `RandomApply` |
| Curve | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| アウトライン | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| Subpath | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| 幾何変換 | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| 出力 | `RenderBitmap` |

`RenderBitmap` は各 `Outline` を通常の `uint8` テンソルに変えます。これらが
`GlyphData` や `GlyphIdData` の中にある場合も、参照、Location、Target は変換後の Payload と
ともに維持されます。

### レンダリングしたグリフを TorchVision で使う

`RenderBitmap` はグレースケールの通常の `H x W` テンソルを返します。
`torchvision.transforms.v2.ToImage()` を画像パイプラインへの境界として使うと、チャンネル次元が追加され、
形状が `1 x H x W` の `tv_tensors.Image` になります。外側の `GlyphData` の Field も
維持されます。
`RenderBitmap(antialias=False)` は Edge Coverage を二値化します。これは Vector の
Rasterization を制御するもので、後段の `v2.Resize` における画像の Resampling を制御する
`antialias` Option とは独立しています。

```python
import torch
from torchvision.transforms import v2

from torchfont.transforms import Compose, LoadGlyph, RenderBitmap

transform = Compose(
    [
        LoadGlyph(),
        RenderBitmap(size=96),
        v2.ToImage(),
        v2.RandomAffine(degrees=(-5.0, 5.0), translate=(0.05, 0.05), fill=0),
        v2.Resize((64, 64), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToPureTensor(),
    ]
)

data = transform(sample)
image = data.data  # Tensor, (1, 64, 64), float32, range [0, 1]
```

幾何変換まではビットマップを `uint8` に保ち、モデルへ渡す直前に
`ToDtype(torch.float32, scale=True)` で変換します。`ToImage()` 自体はピクセル値を
スケーリングしません。`ToPureTensor()` はモデルへ渡す前に `Image` サブクラスを取り除きます。
TorchVision は任意の統合先であり、TorchFont のレンダラーには不要です。

## Functional API

決定論的な処理は `torchfont.transforms.functional` から利用できます。

```python
from torchfont import Outline
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)
outline = F.remove_overlaps(outline)
outline = F.quad_to_cubic(outline, merge_curves=True)
outline = F.affine(outline, angle=10.0)
bitmap = F.render_bitmap(outline, size=64)
shape = bitmap.shape
```

Functional API は乱数を生成しません。ランダムな選択とパラメーターのサンプリングは
`Random*` Transform クラスの責務です。

### 単一グリフのみを扱う

この節の各処理は単一グリフを対象とします。バッチ化された `Outline` を渡すと
エラーになります。

```python
F.affine(batch, angle=5.0)
# ValueError: affine operates on a single outline, got batch shape (64,);
#             iterate with unpad_outlines() first
```

Transform は Collate の前にサンプルごとに実行されます。パイプラインの出力は
[`pad_outlines`](./core-types.md#pad-outlines) または `DataLoader` でバッチ化してください。

### 微分可能性

勾配への対応は処理ごとに異なります。

| カーネル | 微分可能 |
| --- | --- |
| `affine` | はい |
| `coord_jitter` | はい。Outline と Noise の両方について |
| `horizontal_flip`, `vertical_flip` | `preserve_winding=False` のときのみ |
| `quad_to_cubic`, `cubic_to_quad`, `merge_curves`, `split_segments` | いいえ |
| `remove_overlaps`, `remove_overlap_groups` | いいえ |
| `normalize_subpath_start_points`, `set_subpath_start_points`, `reorder_subpaths` | いいえ |
| `render_bitmap` | いいえ |

「いいえ」の処理に勾配を要求する Outline を渡すとエラーになります。

```python
outline.coords.requires_grad_()
F.remove_overlaps(outline)
# RuntimeError が発生
```

`affine` と Flip は Tight Bounding Box の中心を軸に変換します。勾配は変換後の
座標を通って流れますが、この中心を通っては流れません。

### デバイス

`LoadGlyph` は CPU の `float32` Outline を返します。`Affine`、Flip、Curve、Overlap、
Subpath の各 Transform と `RenderBitmap` は、CPU の `float32` Outline を必要とします。
それ以外の Outline は呼び出す前に明示的に変換してください。

```python
outline = outline.to("cpu", torch.float32)
```

`RandomCoordJitter` と `functional.coord_jitter` は入力の Device と浮動小数点 dtype を
維持します。

### `torch.compile`

完全な使用例、Outline の動的な長さ、対応範囲については、
[Transform パイプラインのコンパイル](../guide/advanced/torch-compile.md) を参照してください。
