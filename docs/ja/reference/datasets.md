# データセット

TorchFont はローカルのフォントディレクトリを 2 通りにインデックスします。`CodepointDataset` は
各フォントフェイスと、そのフェイスが収録する各 Unicode コードポイントを 1 要素として数えます。
`GlyphIdDataset` は各フォントフェイスと、そのフェイスがアウトラインを持つ各グリフを 1 要素として
数え、どのコードポイントからも辿れないグリフも含めます。どちらのインデックスも決定的で、
アウトラインをロードするときに `transform` がバリエーション位置を選択します。

| データセット | 1 要素の単位 | サンプル | フィルター |
|---|---|---|---|
| `CodepointDataset` | フェイスとコードポイント | `GlyphSample` | `codepoints`, `max_length`, `patterns` |
| `GlyphIdDataset` | フェイスとグリフ ID | `GlyphIdSample` | `max_length`, `patterns` |

`transform` を指定しない場合、`dataset[i]` は `pickle` 可能なサンプルを返します。

| 型 | フィールド |
|---|---|
| `FontRef` | `path: str`, `ttc_index: int` |
| `GlyphRef` | `font: FontRef`, `glyph_id: int` |
| `GlyphSample` | `ref: GlyphRef`, `codepoint: int`, `font_idx: int`, `character_idx: int` |
| `GlyphIdSample` | `ref: GlyphRef`, `font_idx: int` |

## `CodepointDataset`

```python
CodepointDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    max_length: SupportsIndex | None = None,
    patterns: str | Sequence[str] | None = None,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

```python
from torchfont.datasets import CodepointDataset

dataset = CodepointDataset(
    root="data/google/fonts",
    codepoints=[0x41, 0x42],
    patterns="ofl/*/*.ttf",
)
```

各要素は、1 つのフェイスと、その `cmap` がアウトライングリフへ対応付けるコードポイント 1 つの
組です。リガチャや異体字など、どのコードポイントからも辿れないグリフはこのインデックスに
含まれません。

プロパティ:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`
- `character_targets -> LongTensor (N,)`

サンプリング分布は各フェイスが収録するコードポイント数に比例します。異なる分布が必要な用途では
PyTorch のサンプラーで学習時の重みを調整してください。

## `GlyphIdDataset`

```python
GlyphIdDataset(
    root: Path | str,
    *,
    max_length: SupportsIndex | None = None,
    patterns: str | Sequence[str] | None = None,
    transform: Callable[[GlyphIdSample], T] | None = None,
)
```

```python
from torchfont.datasets import GlyphIdDataset

dataset = GlyphIdDataset(
    root="data/google/fonts",
    patterns="ofl/*/*.ttf",
)
```

各要素は、1 つのフェイスと、そのフェイスがアウトラインを持つグリフ 1 つの組です。`.notdef` から
始まり、グリフ ID の昇順に並びます。リガチャや異体字など、OpenType の機能で置換されるグリフにも
ここから到達できます。

グリフ ID はフェイスごとのローカルな値です。そのため `codepoints` フィルターは持たず、サンプルは
コードポイントや文字のターゲットを持ちません。

プロパティ:

- `font_classes -> list[FontRef]`
- `font_targets -> LongTensor (N,)`
- `glyph_ids -> LongTensor (N,)`

`glyph_ids` は各サンプルのフェイスローカルなグリフ ID です。フェイスをまたいで比較できる値では
ないため、学習ターゲットではなくサンプル選択のためのデータとして使ってください。

サンプリング分布は各フェイスが収録するアウトライングリフ数に比例します。

## 系列長のフィルター

どちらのデータセットも `max_length` を受け取ります。アウトラインの要素数が指定値以下の
グリフだけを残します。

```python
dataset = GlyphIdDataset(root="data/google/fonts", max_length=128)
```

条件を満たすグリフが 1 つも残らないフェイスは、他のフィルターと同様に除外されます。

## グリフのロード

インデックスと未変換の Sample は決定的です。各 Face の Default Location を読むには
`LoadGlyph()`、変換のたびに位置を 1 点抽出するには `location="random"` を指定します。
Static Face では両方の Policy が空の位置を使うため、同じ Outline になります。

```python
from torchfont.transforms import LoadGlyph

evaluation = CodepointDataset(root, transform=LoadGlyph())
training = CodepointDataset(root, transform=LoadGlyph(location="random"))
```

`LoadGlyph` は `GlyphSample` に対しては `GlyphData` を、`GlyphIdSample` に対しては
`GlyphIdData` を返します。どちらもロードしたペイロード、グリフ参照、解決済みの位置、連続値の
ターゲットを持ち、コードポイントのターゲットを持つのは `GlyphData` だけです。
[基本型](./core-types.md) を参照してください。

## 明示的な位置のロード

決定的な再現には関数形式 API を使えます。

```python
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)  # face default
outline = F.load_glyph(sample.ref, {"wght": 700.0})
```
