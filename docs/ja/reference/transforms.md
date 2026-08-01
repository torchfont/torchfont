# Transform

TorchFont の transform は torchvision v2 と同様に、意味を持つデータ型、
クラスベースの確率的 transform、決定論的 functional kernel、pytree の合成を
基本にします。

## データ型

```python
from torchfont.transforms import Bitmap, GlyphData, Outline, OutlinePatches, TFTensor
```

`Outline(types, coords)` は不可分な二つの tensor を一つにまとめます。
`TFTensor` は torchvision の `TVTensor` に対応する意味 tensor の基底であり、
`Bitmap(tensor)` は rasterized glyph を表す subclass です。通常の tensor 演算結果は
plain tensor に戻るため、意味型の dispatch が model 内部まで伝播しません。copy、
device・dtype 変換では subclass を維持します。
`GlyphData[T]` は変換中の payload、元の dataset sample、実際に使用した variation
location を保持します。payload は generic なので、メタデータを失わずに
`Outline` から `OutlinePatches` や bitmap へ変換できます。

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
torchvision と同様に、`Compose` と `RandomApply` の callable 列は学習可能な child
module として登録されません。したがって transform pipeline は parameter や可変な
runtime state ではなく、設定を保持するものとして扱います。

torchvision v2 と同様に、transform と container は一つの pytree または複数の
位置引数を受け取れます。leaf 間の関係を parameter sampling 前に確認する
custom transform のために `check_inputs()` を利用できます。`Compose` と
`RandomApply` には空でない callable の列を渡します。
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
| outline | `RemoveOverlaps`, `RandomRemoveOverlaps`, `Patchify` |
| subpath | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| 幾何変換 | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| 出力 | `RenderBitmap` |

`Patchify` は各 `Outline` を `OutlinePatches` に変えます。`RenderBitmap` は
`uint8` tensor を持つ `Bitmap` に変えます。これらが `GlyphData` 内にある場合も、sample と
location は変換後の payload とともに維持されます。

## Functional kernel

決定論的な処理は `torchfont.transforms.functional` から利用できます。

```python
from torchfont.transforms import Outline
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

functional は意味的な入力型により kernel を dispatch します。下流の
`Outline` subclass は transform class を変更せず functional を特殊化できます。

```python
@F.register_kernel(F.horizontal_flip, CustomOutline)
def horizontal_flip_custom(inpt, *, preserve_winding=True): ...
```

組み込み outline transform は `Outline` instance を選択するため、登録した kernel は
`Outline` subclass の dispatch を拡張します。独自の意味型には、その型を
`_transformed_types` で選択する custom transform も必要です。複数 leaf の関係に応じて
選択する場合のみ `_needs_transform_list()` を override します。
