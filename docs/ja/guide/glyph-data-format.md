# グリフデータ形式

<!-- markdownlint-disable MD013 -->

## サンプルを取得する

前の章で作成したデータセットからサンプルを取得します。次のコードを実行してください。

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
outline = data.data
types, coords = outline.types, outline.coords

print(data.ref)  # Glyph 参照
print(types)  # 要素型の系列
print(coords)  # 座標の系列
print(data.font_idx)  # Font Face の Class ID
print(data.character_idx)  # Character の Class ID
print(data.weight)  # OpenType Weight
print(data.width)  # OpenType Width のパーセント値
print(data.italic)  # OpenType Italic 値
print(data.slant)  # Slant 角度
print(data.optical_size)  # ポイント単位の Optical Size
```

返り値の `GlyphData[Outline]` は、意味型の Outline、決定的な Glyph 参照、Dataset
固有の Target を浅い一つの Record に保持します。

## アウトラインモデル

グリフのアウトラインは、パス要素の系列として表現されます。

- **パス要素**: 1 つの要素型と 1 行の座標からなる最小単位
- **サブパス**: グリフを構成する一続きの曲線ひとつを表すパス要素の系列
- **アウトライン**: グリフ 1 文字分の輪郭を表すパス要素の系列

`types` は要素型を整数で並べた `(seq_len,)` の `LongTensor`、
`coords` は座標を並べた `(seq_len, 6)` の `FloatTensor` です。

## 要素型

要素型は `ElementType` で定義されています。次のコードで値と名前の対応を確認できます。

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
- `ElementType.PAD` は `pad_sequence` や独自のパディングで出現します

## 座標

各パス要素の座標は 6 次元のベクトルです。次のコードで形状と内容を確認できます。

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

使用する次元は要素型によって異なります。

- **`MoveTo` / `LineTo`**: 終点 `(x, y)` のみ使用。制御点は 0
- **`QuadTo`**: 制御点 `(cx0, cy0)` と終点 `(x, y)` を使用。`cx1`、`cy1` は 0
- **`CurveTo`**: 制御点 `(cx0, cy0)`、`(cx1, cy1)` と終点 `(x, y)` をすべて使用
- **`Close` / `End` / `Pad`**: すべて 0

::: info
座標は `em` 単位です。フォントのデザイン単位を `unitsPerEm` で
割った値として表されます。
:::

2 次ベジェは 3 次ベジェへの変換をせず `QuadTo` としてそのまま出力されます。テンソル形状を固定するため、`QuadTo` の座標は `[cx0, cy0, 0, 0, x, y]` です。

## フォントフェイスと文字のラベル

### `font_idx`

`font_idx` はフォントフェイスのクラス ID です。次のコードで永続的なフェイス参照を確認できます。

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
