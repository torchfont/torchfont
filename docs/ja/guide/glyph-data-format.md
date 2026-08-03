# グリフデータ形式

<!-- markdownlint-disable MD013 -->

## サンプルを取得する

前のチャプターで作成した Dataset からサンプルを取得します。次のコードを実行してください。

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
    transform=LoadGlyph(),
)

data = dataset[0]
sample = data.sample
outline = data.data
types, coords = outline.types, outline.coords

print(sample.ref)  # グリフ参照
print(types)  # Element Type の系列
print(coords)  # Coordinates の系列
print(sample.font_idx)  # font faceのクラスID
print(sample.character_idx)  # 文字のクラス ID
```

返り値は `GlyphData[Outline]` です。`data` が意味型outlineを、`sample` が
決定的なグリフ参照とdataset-localなtarget indexを保持します。

## Outline モデル

グリフのアウトラインは、Path Element の系列として表現されます。

- **Path element**: 1 つの Element Type と 1 行の Coordinates からなる最小単位
- **Subpath**: グリフを構成する一続きの曲線ひとつを表す Path Element の系列
- **Outline**: グリフ 1 文字分の輪郭を表す Path Element の系列

`types` は Element Type を整数で並べた `(seq_len,)` の `LongTensor`、
`coords` は Coordinates を並べた `(seq_len, 6)` の `FloatTensor` です。

## Element Type

Element Type は `ElementType` で定義されています。次のコードで値と名前の対応を確認できます。

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph
from torchfont.structures import ElementType

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
    transform=LoadGlyph(),
)

outline = dataset[0].data
types, coords = outline.types, outline.coords

print(types)
print(ElementType(types[0].item()).name)
```

実行すると次のような出力が得られます。

```
tensor([1, 2, 3, ..., 5, 6])
MOVE_TO
```

種類は `MoveTo`、`LineTo`、`QuadTo`、`CurveTo`、`Close`、`End`、`Pad` の 7 つです。

- `ElementType.END` はシーケンス終端を表します
- `ElementType.PAD` は `pad_sequence` や独自 Padding で出現します

## Coordinates

各 Path Element の Coordinates は 6 次元のベクトルです。次のコードで形状と内容を確認できます。

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    codepoints=range(0x41, 0x5B),
    transform=LoadGlyph(),
)

outline = dataset[0].data
types, coords = outline.types, outline.coords

print(coords.shape)
print(coords[0])
```

実行すると次のような出力が得られます。

```
torch.Size([seq_len, 6])
tensor([cx0, cy0, cx1, cy1, x, y])
```

使用する次元は Element Type によって異なります。

- **`MoveTo` / `LineTo`**: 終点 `(x, y)` のみ使用。制御点は 0
- **`QuadTo`**: 制御点 `(cx0, cy0)` と終点 `(x, y)` を使用。`cx1`、`cy1` は 0
- **`CurveTo`**: 制御点 `(cx0, cy0)`、`(cx1, cy1)` と終点 `(x, y)` をすべて使用
- **`Close` / `End` / `Pad`**: すべて 0

::: info
Coordinates は em 単位です。フォントの design units を `unitsPerEm` で
割った値として表されます。
:::

2 次ベジェは 3 次ベジェへの変換をせず `QuadTo` としてそのまま出力されます。テンソル形状を固定するため、`QuadTo` の Coordinates は `[cx0, cy0, 0, 0, x, y]` です。

## Font faceと文字のラベル

### `font_idx`

`font_idx`はfont faceのクラスIDです。次のコードで永続的なface参照を確認できます。

```python
from torchfont.datasets import GlyphDataset

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

sample = dataset[0]

print(dataset.font_classes[sample.font_idx])
```

実行すると次のような出力が得られます。

```
FontRef(path='.../Aclonica-Regular.ttf', ttc_index=0)
```

### `character_idx`

`character_idx` は文字のクラス ID です。`character_classes` から対応する文字を取得できます。次のコードで確認できます。

```python
from torchfont.datasets import GlyphDataset

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

sample = dataset[0]

print(dataset.character_classes[sample.character_idx])
```

実行すると次のような出力が得られます。

```
A
```
