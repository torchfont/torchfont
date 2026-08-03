# Transform

TorchFont の transform は torchvision v2 と同様に、意味を持つデータ型、
クラスベースの確率的 transform、決定論的 functional kernel、pytree の合成を
基本にします。

## データ型

```python
from torchfont import tf_tensors
from torchfont.structures import GlyphData, Outline
```

`Outline(types, coords)` は不可分な二つの tensor を一つにまとめます。
`tf_tensors.TFTensor` は torchvision の `TVTensor` に対応する意味 tensor の基底であり、
`tf_tensors.Bitmap(tensor)` は rasterized glyph を表す subclass です。通常の tensor
演算結果はplain tensorに戻るため、意味型のdispatchがmodel内部まで伝播しません。
torchvisionの`TVTensor`と同様、`clone()`、`detach()`、`pin_memory()`、
`requires_grad_()`、`to()`ではsubclassを維持します。一方、`float()`や`cpu()`などの
convenience methodは通常の演算と同じ規則に従い、plain tensorを返します。storageを
copyせず意味型へ戻すには`tf_tensors.wrap(tensor, like=bitmap)`を使います。
`GlyphData[T]` は変換中の payload、元の dataset sample、実際に使用した variation
location を保持します。payload は generic なので、メタデータを失わずに
`Outline` から bitmap へ変換できます。

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
        RandomApply([RandomSplitSegments(split_probability=0.2)], p=0.5),
        RemoveOverlaps(),
    ]
)

data = transform(sample)
outline = data.data
```

`LoadGlyph` は `GlyphSample` または `GlyphRef` を受け取ります。sample は
`GlyphData[Outline]` に、ref 単体は `Outline` になります。
`RandomLocation` は `VariableGlyphSample` に対応し、サンプリングした location を
`GlyphData.location` に保存します。

`Transform` はネストした pytree を flatten し、`make_params()` を一度だけ呼び、
一致するすべての意味的な leaf に同じパラメータを適用して元の構造を復元します。
このため対応する複数 outline には同じ flip、affine parameter、element 単位の乱数が
適用されます。確率的 transform は PyTorch の default RNG を使うため、
`torch.manual_seed` と DataLoader worker の seed が通常どおり機能します。
組み込み transform は設定のみを保持し、pickle 可能です。container に渡す custom
callable が pickle 可能かどうかは、その callable 自体に依存します。
torchvision と同様に、通常の callable 列は学習可能な child module として
登録されません。module 登録が必要な場合、`Compose` と `RandomApply` には
`torch.nn.ModuleList` を渡せます。

torchvision v2 と同様に、transform と container は一つの pytree または複数の
位置引数を受け取れます。leaf 間の関係を parameter sampling 前に確認する
custom transform のために `check_inputs()` を利用できます。`Compose` と
`RandomApply` には空でない callable の列または
`torch.nn.ModuleList` を渡します。
一回の呼び出しでは、一致した leaf 間で parameter を共有します。独立した乱数を
使うべき sample には transform を個別に呼び出します。

`RandomApply(transforms, p)` は transform 列全体を適用するか制御します。
`RandomSplitSegments.split_probability` などは、適用された transform 内部の挙動を
制御します。

## 組み込み Transform

| 分類 | Transform |
| --- | --- |
| 読み込み | `LoadGlyph`, `RandomLocation` |
| コンテナ | `Compose`, `RandomApply` |
| curve | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| outline | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| subpath | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| 幾何変換 | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| 出力 | `RenderBitmap`, `ToPureTensor` |

`RenderBitmap` は各 `Outline` を `uint8` tensor を持つ `Bitmap` に変えます。
これらが `GlyphData` 内にある場合も、sample と location は変換後の payload とともに
維持されます。`ToPureTensor` はmodelへ入力する前に、tensor storageをcopyせず
`TFTensor` subclassを取り除きます。

## Functional kernel

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

functional API は乱数を生成しません。ランダムな選択と parameter sampling は
`Random*` transform class の責務です。

functional は意味的な入力型により内部 kernel を dispatch します。
非対応の入力型には `TypeError` を送出します。
