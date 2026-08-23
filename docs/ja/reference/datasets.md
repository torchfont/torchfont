# データセット

TorchFont のインデックス規則は一つです。各フォントフェイスと、そのフェイスが収録する各 Unicode
コードポイントを 1 要素として数えます。アウトラインをロードするときに `transform` がバリエーション
位置を選択します。

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/google/fonts",
    codepoints=[0x41, 0x42],
    patterns="ofl/*/*.ttf",
)
```

`transform` を指定しない場合、`dataset[i]` は `pickle` 可能な `GlyphSample` を返します。

| 型 | フィールド |
|---|---|
| `FontRef` | `path: str`, `ttc_index: int` |
| `GlyphRef` | `font: FontRef`, `glyph_id: int` |
| `GlyphSample` | `ref: GlyphRef`, `codepoint: int`, `font_idx: int`, `character_idx: int` |

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

インデックスと未変換の Sample は決定的です。各 Face の Default Location を読むには
`LoadGlyph()`、変換のたびに位置を 1 点抽出するには `location="random"` を指定します。
Static Face では両方の Policy が空の位置を使うため、同じ Outline になります。

```python
from torchfont.transforms import LoadGlyph

evaluation = GlyphDataset(root, transform=LoadGlyph())
training = GlyphDataset(root, transform=LoadGlyph(location="random"))
```

プロパティ:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`
- `character_targets -> LongTensor (N,)`

サンプリング分布は各フェイスが収録するコードポイント数に比例します。異なる分布が必要な用途では
PyTorch のサンプラーで学習時の重みを調整してください。

## 明示的な位置のロード

決定的な再現には関数形式 API を使えます。

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)  # face default
outline = F.load_glyph(sample.ref, {"wght": 700.0})
```
