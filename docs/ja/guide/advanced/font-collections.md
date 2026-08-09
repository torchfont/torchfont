# フォントコレクション

TrueType Collection（`.ttc`）と OpenType Collection（`.otc`）は、1 つのファイルに
複数のフォントフェイスを格納します。`GlyphDataset` はコレクション内のすべてのフェイスを
自動的にインデックスへ登録します。

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(
    root="data/fonts",
    patterns=("**/*.ttc", "**/*.otc"),
    transform=LoadGlyph(),
)

for font in dataset.font_classes:
    print(font.path, font.ttc_index)
```

各フェイスには個別の `font_idx` が割り当てられます。`FontRef.ttc_index` はファイル内の
フェイスを 0 から始まる番号で表します。通常の単一フェイスフォントでは 0 です。
それ以外の扱いは、データセット内のほかのフォントと同じです。

## フェイスを直接選択する

ファイルとフェイスインデックスが分かっている場合は `FontRef` を使用します。

```python
from torchfont import FontRef, GlyphRef
from torchfont.transforms import LoadGlyph

ref = GlyphRef(
    font=FontRef("data/fonts/example.ttc", ttc_index=2),
    codepoint=ord("A"),
)
outline = LoadGlyph()(ref)
```

OpenType Collection にはバリアブルフォントのフェイスを含めることもできます。フェイスと
バリエーション位置は独立して選択されます。`ttc_index` でフェイスを選び、そのフェイスの
バリエーション軸を `LoadGlyph(location="random")` でサンプリングできます。

位置のサンプリングと明示的な軸の値については、
[バリアブルフォント](./variable-fonts.md) を参照してください。
