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
        RandomApply(RandomSplitSegments(split_probability=0.2), p=0.5),
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

`Transform` はネストした pytree を flatten し、一致する意味的な leaf ごとに独立して
parameter を生成し、元の構造を復元します。同じ variable glyph の互換な instance など、
対応する複数 outline に同じ flip、affine parameter、element 単位の乱数を適用する場合は
transform を `SameParams` で包みます。確率的 transform は PyTorch の default RNG を使うため、
`torch.manual_seed` と DataLoader worker の seed が通常どおり機能します。
`SameParams(RandomLocation())` を使うと同じfontの複数glyphに一つのlocationを選べますが、
異なるfont間でraw axis valueを共有することは拒否します。
組み込み transform は設定のみを保持し、pickle 可能です。`Compose` に通常の list を
渡した場合も、子transformは内部の`torch.nn.ModuleList`に登録されます。plain callableは
意図的に対応しません。小さな`nn.Module`を定義し、挙動、表示、pickle要件を明示します。
これにより torchvision の歴史的な未登録 callable list の挙動を引き継がず、module
traversal、state dictionary、hook、設定表示を PyTorch の規則に揃えます。

torchvision v2 と同様に、transform と container は一つの pytree または複数の
位置引数を受け取れます。leaf 間の関係を parameter sampling 前に確認する
custom transform のために `check_inputs()` を利用できます。`Compose` には
`nn.Module` の列または `nn.ModuleList` を渡し、空の列は identity transform として
扱います。`RandomApply` は一つの
`nn.Module`、`SameParams` は一つの `Transform` を包みます。複数transformを
`RandomApply` でまとめる場合は、内側に `Compose` を置きます。

`RandomApply(transform, p)` は一つの transform を適用するか制御します。
`RandomSplitSegments.split_probability` などは、適用された transform 内部の挙動を
制御します。

## 組み込み Transform

| 分類 | Transform |
| --- | --- |
| 読み込み | `LoadGlyph`, `RandomLocation` |
| コンテナ | `Compose`, `RandomApply`, `SameParams` |
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

これらの transform が `nn.Module` なのは、合成、module 登録、PyTorch RNG、pytree
処理のためです。Rust-backed outline functional は CPU/NumPy 境界を通る前処理であり、
autograd への参加、accelerator 上での実行維持、`torch.compile` による capture は
保証しません。現在の functional は signature に記載した単一の意味型を処理します。
複数の outline 表現が必要になるまでは kernel registry を導入しません。
