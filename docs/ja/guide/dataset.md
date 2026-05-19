# GlyphDataset による読み込み

## データセットを読み込む

Google Fonts のセットアップが完了したら、`GlyphDataset` を使って読み込みます。
次のコードを実行してください。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="data/google/fonts")

print(f"{len(dataset)=}")
print(f"{len(dataset.style_classes)=}")
print(f"{len(dataset.content_classes)=}")
```

実行すると次のような出力が得られます。
`style_classes` はフォントスタイルの種類数、`content_classes` はデータセットに含まれるユニークなコードポイントの数です。

```
len(dataset)=13745683
len(dataset.style_classes)=9113
len(dataset.content_classes)=1112000
```

## `patterns` で絞り込む

`patterns` を指定することで、読み込むフォントファイルを絞り込めます。
Google Fonts では次の patterns を推奨します。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
)
```

各パターンの意図は次のとおりです。

- **`apache/*/*.ttf`、`ofl/*/*.ttf`、`ufl/*/*.ttf`**: Google Fonts のライセンスディレクトリは
  この3つのみです。リポジトリにはこれら以外にもディレクトリがあり、テスト用フォントなどが
  含まれることがあるため、明示的に3ディレクトリだけを対象にします。
- **`/*/*.ttf`（`/**/*.ttf` ではない）**: Google Fonts の配布形式は TTF のみです。
  また、`/**/*.ttf` にするとサブディレクトリに置かれた余分なフォントが
  含まれてしまうため、1階層のみを対象にします。
- **`!ofl/adobeblank/AdobeBlank-Regular.ttf`**: AdobeBlank はフォント表示のフォールバック
  検証用に設計されたフォントで、対応するコードポイント数が極めて多い一方、
  グリフに実質的な字形データが含まれていません。機械学習のデータセットに混入すると
  品質を著しく低下させるため、除外することが望ましいです。

このコードを実行すると次のような出力が得られます。patterns を指定しない場合と比べて、
スタイル数とコンテンツクラス数（コードポイント数）が絞られていることが確認できます。

```
len(dataset)=12460609
len(dataset.style_classes)=8951
len(dataset.content_classes)=114254
```

### コードポイントを絞る

`codepoints` を指定すると、インデックス時に対象外のコードポイントを除外できます。

アルファベット26文字（A–Z）に絞るには、次のように `codepoints` 引数を追加して実行してください。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
)
```

実行結果は次のとおりです。

```
len(dataset)=220939
len(dataset.style_classes)=8498
len(dataset.content_classes)=26
```

Unicode の基本多言語面（BMP、U+0000–U+FFFF）に絞るには、次のように指定してください。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x0000, 0x10000),
)
```

実行結果は次のとおりです。

```
len(dataset)=12027967
len(dataset.style_classes)=8951
len(dataset.content_classes)=60004
```
