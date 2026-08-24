# バリアブルフォント

バリアブルフォントは、ウェイト（`wght`）、幅（`wdth`）、オプティカルサイズ（`opsz`）
などの軸を持ち、関連する複数のデザインを 1 つのフォントで表現します。`LoadGlyph` は、
各グリフを読み込むときに軸上の位置を決定します。

再現可能な評価にはデフォルト位置を使用します。

```python
from torchfont.datasets import CodepointDataset
from torchfont.transforms import LoadGlyph

evaluation = CodepointDataset(
    root="data/fonts",
    patterns=("**/*.ttf", "**/*.otf", "**/*.ttc", "**/*.otc"),
    transform=LoadGlyph(),
)
```

`location="random"` を指定すると、アクセスするたびに各軸の最小値から最大値までを
一様にサンプリングします。個別のインスタンスファイルを用意せず、フォントのデザイン空間を
学習時のデータ拡張として利用できます。

```python
training = CodepointDataset(
    root="data/fonts",
    transform=LoadGlyph(location="random"),
)

sample = training[0]
print(sample.location)  # 例: {"wght": 573.2, "opsz": 41.7}
```

サンプリングには PyTorch の乱数生成器が使われるため、DataLoader ワーカーのシードと
`torch.manual_seed` が適用されます。返された `Outline` に実際に使用した軸の値は
`GlyphData.location` に記録されます。静的フォントでは空の辞書となり、ランダム位置と
デフォルト位置では同じ `Outline` が返されます。

`LoadGlyph(location="random")` はすべての軸をサンプリングします。実験で明示的な位置が
必要な場合は `functional.load_glyph(ref, location={...})` を使用します。

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(
    sample.ref,
    location={"wght": 700.0, "wdth": 90.0},
)
```
