# グリフデータ形式

<!-- markdownlint-disable MD013 -->

## サンプルを取得する

前のチャプターで作成した Dataset からサンプルを取得します。次のコードを実行してください。

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

print(sample.types)        # Element Type の系列
print(sample.coords)       # Coordinates の系列
print(sample.style_idx)    # 書体スタイルのクラス ID
print(sample.content_idx)  # 文字内容のクラス ID
```

返り値は `GlyphSample` という Dataclass です。各フィールドには名前でアクセスします。

## Outline モデル

グリフのアウトラインは、Path Element の系列として表現されます。

- **Path element**: 1 つの Element Type と 1 行の Coordinates からなる最小単位
- **Subpath**: グリフを構成する一続きの曲線ひとつを表す Path Element の系列
- **Outline**: グリフ 1 文字分の輪郭を表す Path Element の系列

`sample.types` は Element Type を整数で並べた `(seq_len,)` の `LongTensor`、`sample.coords` は Coordinates を並べた `(seq_len, 6)` の `FloatTensor` です。

## Element Type

Element Type は `ElementType` で定義されています。次のコードで値と名前の対応を確認できます。

```python
from torchfont.datasets import GlyphDataset
from torchfont.io import ElementType

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

print(sample.types)
print(ElementType(sample.types[0].item()).name)
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

print(sample.coords.shape)
print(sample.coords[0])
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
Coordinates はフォントの `units_per_em` で正規化されています。
:::

2 次ベジェは 3 次ベジェへの変換をせず `QuadTo` としてそのまま出力されます。テンソル形状を固定するため、`QuadTo` の Coordinates は `[cx0, cy0, 0, 0, x, y]` です。

## スタイルとコンテンツのラベル

### `style_idx`

`style_idx` は書体スタイルのクラス ID です。次のコードで対応するスタイル名を確認できます。

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

print(dataset.style_classes[sample.style_idx])
```

実行すると次のような出力が得られます。

```
Aclonica Regular
```

### `content_idx`

`content_idx` は文字内容のクラス ID です。`content_classes` から対応する文字を取得できます。次のコードで確認できます。

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

print(dataset.content_classes[sample.content_idx])
```

実行すると次のような出力が得られます。

```
A
```
