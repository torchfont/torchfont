# GlyphDataset の構築

`GlyphDataset`へローカルfont directoryを渡します。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="data/google/fonts")

print(len(dataset))
print(len(dataset.font_classes))
print(len(dataset.character_classes))
```

各要素はfont faceと、そのfaceが収録するcodepoint一つを表します。variable fontを
named instanceやgrid locationで展開しません。variable faceも収録codepointごとに
ちょうど1要素を持ちます。

## ファイルのfilter

`patterns`にはgitignore形式のpattern一つ、またはsequenceを渡せます。

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

## 文字のfilter

整数のUnicode codepointを指定します。

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    codepoints=range(0x41, 0x5B),
)
```

重複codepointは除去され、indexは決定的です。要求されたoutline glyphを一つも持たない
fontは除外されます。

## Variation locationの選択

raw sampleは決定的です。評価時はdefault locationをロードします。

```python
from torchfont.transforms import LoadGlyph

dataset = GlyphDataset(root="data/google/fonts", transform=LoadGlyph())
```

学習時はアクセスごとにlocationを1点抽出できます。

```python
from torchfont.transforms import RandomLocation

dataset = GlyphDataset(root="data/google/fonts", transform=RandomLocation())
```

`RandomLocation`はPyTorch RNGのseedに従います。static faceでは`LoadGlyph`と同じ
空locationになります。
