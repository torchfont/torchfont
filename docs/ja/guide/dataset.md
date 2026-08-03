# `GlyphDataset` の構築

`GlyphDataset` へローカルフォントのディレクトリを渡します。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
print(len(dataset.character_classes))
```

各要素はフォントフェイスと、そのフェイスが収録するコードポイント一つを表します。各フェイスは収録
コードポイントごとに 1 要素を持ちます。

## ファイルのフィルター

`patterns` には `gitignore` 形式のパターン一つ、またはシーケンスを渡せます。

```python
dataset = GlyphDataset(
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
dataset = GlyphDataset(
    root="data/google/fonts",
    codepoints=range(0x41, 0x5B),
)
```

重複コードポイントは除去され、インデックスは決定的です。要求されたアウトライングリフを一つも持たない
フォントは除外されます。

## バリエーション位置の選択

未変換のサンプルは決定的です。評価時はデフォルト位置をロードします。

```python
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(root="data/google/fonts", transform=LoadGlyph())
```

学習時はアクセスごとに位置を 1 点抽出できます。

```python
from torchfont.transforms import RandomLocation

dataset = GlyphDataset(root="data/google/fonts", transform=RandomLocation())
```

`RandomLocation` は PyTorch RNG のシードに従います。静的フェイスでは `LoadGlyph` と同じ
空の位置になります。
