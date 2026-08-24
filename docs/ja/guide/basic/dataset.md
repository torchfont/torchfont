# データセットの構築

`CodepointDataset` へローカルフォントのディレクトリを渡します。

```python
from torchfont.datasets import CodepointDataset

dataset = CodepointDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
print(len(dataset.character_classes))
```

各要素は、フォントフェイスとそのフェイスが収録するコードポイント 1 つの組を表します。

## ファイルのフィルター

`patterns` には `gitignore` 形式のパターン一つ、またはシーケンスを渡せます。

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/*.ttf",
    ),
)
```

## 文字のフィルター

整数の Unicode コードポイントを指定します。

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    codepoints=range(0x41, 0x5B),
)
```

重複したコードポイントは除去され、インデックスは決定的に構築されます。指定した
アウトライングリフを 1 つも持たないフォントは除外されます。

## コードポイントから辿れないグリフの利用

`CodepointDataset` は `cmap` テーブルの対応関係をインデックスするため、リガチャや異体字など
OpenType の機能で置換されるグリフは現れません。フェイスが持つすべてのグリフを対象にするには
`GlyphIdDataset` を使います。

```python
from torchfont.datasets import GlyphIdDataset

dataset = GlyphIdDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
```

各要素は、フォントフェイスとグリフ ID 1 つの組を表します。`.notdef` から始まり、グリフ ID の
昇順に並びます。グリフ ID はフェイスごとのローカルな値なので、`codepoints` フィルターはなく、
サンプルは文字のターゲットを持ちません。`patterns` と `transform`、`LoadGlyph` の使い方は
上と同じです。

## バリエーション位置の選択

未変換のサンプルは決定的です。評価時はデフォルト位置をロードします。

```python
from torchfont.transforms import LoadGlyph

dataset = CodepointDataset(root="data/google/fonts", transform=LoadGlyph())
```

学習時はアクセスごとに位置を 1 点抽出できます。

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    transform=LoadGlyph(location="random"),
)
```

ランダムな位置は PyTorch の乱数生成器のシードに従います。静的フォントではデフォルト位置と
同じ空の辞書になります。

明示的な位置とサンプリングの詳細は、[バリアブルフォント](../advanced/variable-fonts.md) を
参照してください。
