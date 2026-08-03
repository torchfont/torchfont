# Transform

TorchFont の Transform は TorchVision Transforms v2（`torchvision.transforms.v2`）と同様に、意味を持つデータ型、
クラスベースの確率的 Transform、決定論的な Functional カーネル、PyTree の合成を
基本にします。

## データ型

```python
from torchfont.structures import GlyphData, Outline
```

`Outline(types, coords)` は不可分な二つのテンソルを一つにまとめます。
`GlyphData[T]` は変換中の Payload、Glyph 参照、実際に使用した Variation Location、Target
を保持します。Payload は Generic なので、Metadata を失わずに
`Outline` から通常のビットマップテンソルへ変換できます。ラスタライズしたグリフには
TorchFont 固有のテンソルサブクラスを設けず、画像として扱う必要がある境界で TorchVision の
`ToImage()` を明示的に適用します。

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

`LoadGlyph` は `GlyphSample` または `GlyphRef` を受け取ります。サンプルは
`GlyphData[Outline]` に、参照単体は `Outline` になります。
`LoadGlyph` は、`location="random"` を指定しない限り Face の Default Location を使います。
Random Policy は `GlyphSample` または `GlyphRef` に対して位置を 1 点抽出し、
`GlyphData.location` に保存します。Static Face では空の位置になります。
Dataset Sample に対しては、返される `GlyphData` の並列な `weight`、`width`、`italic`、
`slant`、`optical_size` Target も解決します。

`Transform` はネストした PyTree を平坦化し、一致する意味的なリーフごとに独立して
パラメーターを生成し、元の構造を復元します。同じバリアブルフォントの複数グリフを一つの
位置で扱う場合など、対応する複数アウトラインに同じ反転、アフィンパラメーター、要素単位の
乱数を適用する場合は、Transform を `SameParams` で包みます。確率的 Transform は PyTorch の
デフォルト RNG を使うため、`torch.manual_seed` と `DataLoader` ワーカーのシードが通常どおり
機能します。`SameParams(LoadGlyph(location="random"))` を使うと同じ Font の複数 Glyph に
一つの位置を選べますが、異なる Font 間で未変換の Axis 値を共有することは拒否します。
組み込み Transform は設定のみを保持し、`pickle` 可能です。`Compose` に通常のリストを
渡した場合も、子 Transform は内部の `torch.nn.ModuleList` に登録されます。通常の
`callable` には意図的に対応しません。小さな `nn.Module` を定義し、挙動、表示、`pickle`
要件を明示します。これによりモジュールの走査、状態辞書、フック、設定表示を PyTorch の
規則に揃えます。

コンテナーは登録した子をモジュール呼び出し経路で呼ぶため、`forward` フックも機能します。
`train()` と `eval()` は通常どおり伝播しますが、組み込みの確率的 Transform は意図的に
`training` フラグを参照しません。前処理とモデルのモードは別の関心事なので、評価時は
`eval()` でデータ拡張が止まることに依存せず、決定論的パイプラインを明示的に選びます。

`torchvision.transforms.v2` と同様に、Transform とコンテナーは一つの PyTree または複数の
位置引数を受け取れます。リーフ間の関係をパラメーターのサンプリング前に確認する
カスタム Transform のために `check_inputs()` を利用できます。`Compose` は
`nn.Module` の `Iterable` を即座に `nn.ModuleList` へ具体化し、空の `Iterable` は
恒等 Transform として扱います。`RandomApply` は一つの
`nn.Module`、`SameParams` は一つの `Transform` を包みます。複数の Transform を
`RandomApply` でまとめる場合は、内側に `Compose` を置きます。

`RandomApply(transform, p)` は一つの Transform を適用するか制御します。
`RandomSplitSegments.split_probability` などは、適用された Transform 内部の挙動を
制御します。

## 組み込み Transform

| 分類 | Transform |
| --- | --- |
| 読み込み | `LoadGlyph` |
| コンテナ | `Compose`, `RandomApply`, `SameParams` |
| Curve | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| アウトライン | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| Subpath | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| 幾何変換 | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| 出力 | `RenderBitmap` |

`RenderBitmap` は各 `Outline` を通常の `uint8` テンソルに変えます。これらが
`GlyphData` 内にある場合も、参照、Location、Target は変換後の Payload とともに維持されます。

### レンダリングしたグリフを TorchVision で使う

`RenderBitmap` はグレースケールの通常の `H x W` テンソルを返します。
`torchvision.transforms.v2.ToImage()` を画像パイプラインへの境界として使うと、チャンネル次元が追加され、
形状が `1 x H x W` の `tv_tensors.Image` になります。両ライブラリが PyTree を処理するため、
外側の `GlyphData` も維持されます。
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

## Functional カーネル

決定論的な処理は `torchfont.transforms.functional` から利用できます。

```python
from torchfont.structures import Outline
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

これらの Transform が `nn.Module` なのは、合成、モジュール登録、PyTorch RNG、PyTree
処理のためです。Rust バックエンドのアウトライン用 Functional は CPU / NumPy 境界を通る
前処理であり、自動微分への参加、アクセラレーター上での実行維持、`torch.compile` による
キャプチャーは保証しません。現在の Functional はシグネチャーに記載した単一の意味型を
処理します。複数のアウトライン表現が必要になるまではカーネルレジストリを導入しません。
