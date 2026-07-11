# データセット API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` は参照優先の PyTorch Dataset API を提供します。Dataset item
は軽量で pickle しやすい dataclass で、outline の読み込みは transform 内で
`load_glyph`([Transform Utilities](./transforms.md) 参照)を明示的に呼びます。

Dataset の index と target は構築時点のフォントファイルから作られますが、glyph
outline は現在のディスク上のファイルから遅延読み込みされます。Dataset object の
lifetime 中にフォントファイルを変更すること、pickle/unpickle 境界をまたいで変更する
ことは unsupported で、sample と label の不整合を起こす可能性があります。

## 参照型

```python
from torchfont.datasets import (
    FontRef,
    GlyphRef,
    GlyphSample,
    VariableGlyphRef,
    VariableGlyphSample,
)
```

| 型 | フィールド |
| -- | ---------- |
| `FontRef` | `path: str`, `ttc_index: int` |
| `GlyphRef` | `font: FontRef`, `codepoint: int`, `location: Mapping[str, float]` |
| `VariableGlyphRef` | `font: FontRef`, `codepoint: int` |
| `GlyphSample` | `ref: GlyphRef`, `font_idx: int`, `style_idx: int`, `character_idx: int` |
| `VariableGlyphSample` | `ref: VariableGlyphRef`, `font_idx: int`, `character_idx: int` |

`ttc_index` は read-fonts/skrifa が TrueType Collection 内のフォント位置に
使っている名前に合わせています。単一フォントのファイルでは `0` です。

## GlyphDataset

```python
from torchfont.datasets import GlyphDataset
from torchfont.variation import named_instances

dataset = GlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),
    patterns=("**/*.ttf",),
    instances=named_instances,
)
```

`GlyphDataset` は固定済み variation location を index に含めます。instance function
は構築時だけ実行され、pickle state には保存されません。`transform` なしでは
`dataset[i]` は `GlyphSample` を返します。

コンストラクタ:

```python
GlyphDataset(
    root: Path | str,
    *,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    instances: InstanceFn = torchfont.variation.named_instances,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

targets:

- `font_targets -> LongTensor (N,)`
- `style_targets -> LongTensor (N,)`
- `character_targets -> LongTensor (N,)`

class 語彙:

- `font_classes -> list[FontRef]`
- `style_classes -> list[str]`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`

## VariableGlyphDataset

```python
from torchfont.datasets import VariableGlyphDataset
from torchfont.variation import named_instance_count

dataset = VariableGlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),
    instance_count=named_instance_count,
)
```

`VariableGlyphDataset` は location を index に含めません。各アクセスで transform が
新しい location をサンプルする training augmentation に向いています。`instance_count`
は各フォントの離散的な多重度だけを決める instance-count function です。静的フォントも通常の
フォントとして含まれます。

コンストラクタ:

```python
VariableGlyphDataset(
    root: Path | str,
    *,
    instance_count: InstanceCountFn = torchfont.variation.named_instance_count,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[VariableGlyphSample], T] | None = None,
)
```

targets:

- `font_targets -> LongTensor (N,)`
- `character_targets -> LongTensor (N,)`

## Variation Functions

```python
from torchfont.variation import (
    default_instance,
    default_instance_count,
    grid_instances,
    grid_instance_count,
    named_instances,
    named_instance_count,
    random_location,
)
```

組み込み関数:

- `named_instances(font)`: fvar named instance を dedupe して返す。なければ default
- `default_instance(font)`: default location 1 つ
- `grid_instances({"wght": 7, "wdth": 3})`: 等間隔の固定 grid。フォントに存在しない軸は無視し、指定されなかった軸は default を使い、静的フォントは default 1 枠
- `random_location(font, generator=None)`: transform 時 sampling 用の location 1 つ
- `named_instance_count(font)`: `named_instances` と同じ多重度
- `default_instance_count(font)`: instance slot 1 つ
- `grid_instance_count({"wght": 7, "wdth": 3})`: `grid_instances` と同じ多重度

ランダム性は任意の `torch.Generator` で管理します。dataset-level seed はありません。

カスタム instance function は 0 個の location を返せます。未知の軸や、正規化後に
重複する location は Dataset 構築時に `ValueError` になります。
