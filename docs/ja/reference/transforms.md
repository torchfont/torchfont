# Transform

TorchFont の Transform は torchvision v2 と同様に、意味を持つデータ型、
クラスベースの確率的 Transform、決定論的な Functional カーネル、PyTree の合成を
基本にします。

## データ型

```python
from torchfont import tf_tensors
from torchfont.structures import GlyphData, Outline
```

`Outline(types, coords)` は不可分な二つのテンソルを一つにまとめます。
`tf_tensors.TFTensor` は torchvision の `TVTensor` に対応する意味テンソルの基底であり、
`tf_tensors.Bitmap(tensor)` はラスタライズしたグリフを表すサブクラスです。通常のテンソル
演算結果は通常のテンソルに戻るため、意味型のディスパッチがモデル内部まで伝播しません。
torchvision の `TVTensor` と同様、`clone()`、`detach()`、`pin_memory()`、
`requires_grad_()`、`to()` ではサブクラスを維持します。一方、`float()` や `cpu()` などの
簡易メソッドは通常の演算と同じ規則に従い、通常のテンソルを返します。ストレージを
コピーせず意味型へ戻すには `tf_tensors.wrap(tensor, like=bitmap)` を使います。
メタデータを持つカスタム `TFTensor` サブクラスは `wrap()` クラスメソッドをオーバーライドできます。
コピー系の演算と公開 `tf_tensors.wrap()` ヘルパーはグローバルレジストリを使わず、その
メタデータを維持します。
基底実装はメタデータのキーワード引数を黙って無視せず拒否します。`Bitmap` は 2 次元以上を要求し、
2 次元のグレースケールグリフはチャンネル次元を追加せず `H x W` のまま保持します。
`GlyphData[T]` は変換中のペイロード、元のデータセットサンプル、実際に使用したバリエーション
位置を保持します。ペイロードはジェネリックなので、メタデータを失わずに
`Outline` からビットマップへ変換できます。

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
`LoadGlyph` はフェイスのデフォルト位置を使います。`RandomLocation` は `GlyphSample`
または `GlyphRef` に対して位置を 1 点抽出し、`GlyphData.location` に保存します。
静的フェイスでは空の位置になります。

`Transform` はネストした PyTree を平坦化し、一致する意味的なリーフごとに独立して
パラメーターを生成し、元の構造を復元します。同じバリアブルフォントの複数グリフを一つの
位置で扱う場合など、対応する複数アウトラインに同じ反転、アフィンパラメーター、要素単位の
乱数を適用する場合は、Transform を `SameParams` で包みます。確率的 Transform は PyTorch の
デフォルト RNG を使うため、`torch.manual_seed` と `DataLoader` ワーカーのシードが通常どおり
機能します。`SameParams(RandomLocation())` を使うと同じフォントの複数グリフに一つの位置を
選べますが、異なるフォント間で未変換の軸値を共有することは拒否します。
組み込み Transform は設定のみを保持し、`pickle` 可能です。`Compose` に通常のリストを
渡した場合も、子 Transform は内部の `torch.nn.ModuleList` に登録されます。通常の
`callable` には意図的に対応しません。小さな `nn.Module` を定義し、挙動、表示、`pickle`
要件を明示します。これにより torchvision の歴史的な未登録 `callable` リストの挙動を
引き継がず、モジュールの走査、状態辞書、フック、設定表示を PyTorch の規則に揃えます。

コンテナーは登録した子をモジュール呼び出し経路で呼ぶため、`forward` フックも機能します。
`train()` と `eval()` は通常どおり伝播しますが、組み込みの確率的 Transform は意図的に
`training` フラグを参照しません。前処理とモデルのモードは別の関心事なので、評価時は
`eval()` でデータ拡張が止まることに依存せず、決定論的パイプラインを明示的に選びます。

torchvision v2 と同様に、Transform とコンテナーは一つの PyTree または複数の
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
| 読み込み | `LoadGlyph`, `RandomLocation` |
| コンテナ | `Compose`, `RandomApply`, `SameParams` |
| Curve | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| アウトライン | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| Subpath | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| 幾何変換 | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| 出力 | `RenderBitmap`, `ToPureTensor` |

`RenderBitmap` は各 `Outline` を `uint8` テンソルを持つ `Bitmap` に変えます。
これらが `GlyphData` 内にある場合も、サンプルと位置は変換後のペイロードとともに
維持されます。`ToPureTensor` はモデルへ入力する前に、テンソルストレージをコピーせず
`TFTensor` サブクラスを取り除きます。

### レンダリングしたグリフを torchvision で使う

`RenderBitmap` はグレースケールの `H x W` ビットマップを返します。torchvision v2 の
`ToImage()` を画像パイプラインへの境界として使うと、チャンネル次元が追加され、
形状が `1 x H x W` の `tv_tensors.Image` になります。両ライブラリが PyTree を処理するため、
外側の `GlyphData` も維持されます。

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
torchvision は任意の統合先であり、TorchFont のレンダラーには不要です。

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
