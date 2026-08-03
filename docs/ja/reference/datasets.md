# データセット

TorchFontのindex規則は一つです。各font faceと、そのfaceが収録する各Unicode
codepointを1要素として数えます。outlineをロードするときにtransformがvariation
locationを選択します。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/google/fonts",
    codepoints=[0x41, 0x42],
    patterns="ofl/*/*.ttf",
)
```

transformなしの`dataset[i]`はpickle可能な`GlyphSample`を返します。

| Type | Fields |
|---|---|
| `FontRef` | `path: str`, `ttc_index: int` |
| `GlyphRef` | `font: FontRef`, `codepoint: int` |
| `GlyphSample` | `ref: GlyphRef`, `font_idx: int`, `character_idx: int` |

## `GlyphDataset`

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: str | Sequence[str] | None = None,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

indexとraw sampleは決定的です。各faceのdefault locationを読むには`LoadGlyph()`、
transformのたびにlocationを1点抽出するには`RandomLocation()`を使います。static
faceでは両方とも空locationを使うため同じoutlineになります。

```python
from torchfont.transforms import LoadGlyph, RandomLocation

evaluation = GlyphDataset(root, transform=LoadGlyph())
training = GlyphDataset(root, transform=RandomLocation())
```

properties:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`
- `character_targets -> LongTensor (N,)`

sampling分布は各faceが収録するcodepoint数に比例します。異なる分布が必要な用途では
PyTorch samplerで学習weightを調整してください。

## 明示locationのロード

決定的な再現にはfunctional APIを使えます。

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)  # face default
outline = F.load_glyph(sample.ref, {"wght": 700.0})
```
