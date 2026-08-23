# Transform パイプラインのコンパイル

TorchFont の Functional Transform は、
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html)
でキャプチャできます。グリフを先に読み込み、テンソルの処理だけをコンパイルします。

```python
import torch

from torchfont import Outline
from torchfont.datasets import GlyphDataset
from torchfont.transforms import functional as F

dataset = GlyphDataset("data/fonts", codepoints=[ord("A")])
outline = F.load_glyph(dataset[0].ref)


def pipeline(types: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    outline = Outline(types, coords)
    outline = F.remove_overlaps(outline)
    outline = F.cubic_to_quad(outline)
    outline = F.affine(outline, angle=10.0, scale=1.1)
    outline = F.normalize_subpath_start_points(outline)
    return F.render_bitmap(outline, size=32)


# データ依存の出力 shape を扱う PyTorch 2.5 では必要です。
torch._dynamo.config.capture_dynamic_output_shape_ops = True

compiled_pipeline = torch.compile(pipeline, fullgraph=True)
bitmap = compiled_pipeline(outline.types, outline.coords)

print(bitmap.shape)  # torch.Size([32, 32])
print(bitmap.dtype)  # torch.uint8
```

`fullgraph=True` を指定すると、関数内にグラフブレイクがある場合にコンパイルが失敗する
ため、Transform パイプライン全体がキャプチャされたことを確認できます。TorchFont の
ネイティブ演算は CPU カスタム演算のままです。パイプラインのコンパイルは周囲の
PyTorch グラフをキャプチャしますが、各演算の内部実装を融合するものではありません。

## `Outline` の長さが変わる場合

`remove_overlaps` や `cubic_to_quad` は `Outline` の要素数を変更することがあります。
PyTorch 2.5 では、このようなパイプラインをコンパイルする前に
データ依存の出力 shape のキャプチャを有効にします。

```python
torch._dynamo.config.capture_dynamic_output_shape_ops = True
```

まずは `dynamic` 引数を既定値のまま使用してください。入力の長さが変わると、PyTorch は
より動的なグラフへ再コンパイルできます。長さが大きく異なる `Outline` を同じ
コンパイル済み関数に繰り返し渡すことが分かっている場合は、`dynamic=True` を使用します。

## 制約

- フォントの読み込みとバリアブルフォントの位置選択はコンパイル済み関数の外で行います。
- アウトラインのトポロジー演算とレンダリング演算には、CPU の `float32` 座標と
  `torch.long` の `types` が必要です。
- トポロジーを変更する演算と `render_bitmap` は微分できません。`torch.compile` を
  使用しても Autograd の対応範囲は変わりません。
- コンパイル済み関数内では、引数を明示した Functional Transform を使用します。
  Python 側でのパラメーターサンプリングはコンパイル対象の外で行います。
