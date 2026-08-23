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

print(data.ref)  # グリフ参照
print(types)  # 要素型の系列
print(coords)  # 座標の系列
print(data.codepoint)  # Unicode コードポイント
print(data.font_idx)  # フォントフェイスのクラス ID
print(data.character_idx)  # 文字のクラス ID
print(data.weight)  # OpenType のウェイト
print(data.width)  # OpenType の幅のパーセント値
print(data.italic)  # OpenType のイタリック値
print(data.slant)  # 傾斜角度
print(data.optical_size)  # ポイント単位のオプティカルサイズ
```

返り値の `GlyphData[Outline]` は、意味型の `Outline`、決定的なグリフ参照、データセット
固有のターゲットを 1 つの浅いレコードに保持します。

インデックスは Python の整数、連続ターゲットは浮動小数点数です。フォントに存在しない
連続ターゲットは `None` になります。学習アプリケーション側のローカルな `collate_fn` で、
必要なターゲットだけをテンソルに変換し、`NaN` とマスクの併用など欠損値の表現も
選択できます。

## アウトラインモデル

グリフのアウトラインは、パス要素の系列として表現されます。

- **パス要素**: 1 つの要素型と 1 行の座標からなる最小単位
- **サブパス**: グリフを構成する一続きの曲線ひとつを表すパス要素の系列
- **アウトライン**: グリフ 1 文字分の輪郭を表すパス要素の系列

`types` は要素型を並べた `(seq_len,)` shape、dtype が `torch.long` のテンソルです。
`LoadGlyph` が返す `coords` は、`(seq_len, 6)` shape、dtype が `torch.float32` の
テンソルです。

## 要素型

要素型は `ElementType` で定義されています。次のコードで値と名前の対応を確認できます。

```python
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph
from torchfont import ElementType

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

種類は `MOVE_TO`、`LINE_TO`、`QUAD_TO`、`CURVE_TO`、`CLOSE`、`END`、`PAD` の
7 つです。

- `ElementType.END` はシーケンス終端を表します
- `ElementType.PAD` はバッチのパディングで導入された行を標識します。フォントの
  読み込みでは生成されず、単一グリフには含まれません。値と比較するのではなく
  `outline.padding_mask` を読んでください

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

- **`MOVE_TO` / `LINE_TO`**: 終点 `(x, y)` を使用
- **`QUAD_TO`**: 制御点 `(cx0, cy0)` と終点 `(x, y)` を使用
- **`CURVE_TO`**: 2 つの制御点 `(cx0, cy0)`、`(cx1, cy1)` と終点 `(x, y)` を使用
- **`CLOSE` / `END` / `PAD`**: 使用する座標なし

::: info
座標は `em` 単位です。フォントのデザイン単位を `unitsPerEm` で
割った値として表されます。
:::

2 次ベジェ曲線は 3 次ベジェ曲線へ変換せず、`QUAD_TO` として出力されます。テンソルの
shape を固定するため、すべての要素が 6 個の値を持ちます。未使用の位置にある値には
意味がありません。

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
