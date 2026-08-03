# データセット API

<!-- markdownlint-disable MD013 -->

`torchfont.datasets` は参照優先の PyTorch Dataset API を提供します。Dataset item
は軽量で pickle しやすい dataclass です。outline は dataset の `transform` 引数に
`LoadGlyph()` または合成した transform pipeline を渡して遅延読み込みします
（[Transform Utilities](./transforms.md) 参照）。

Dataset の index と class target は構築時点のフォントファイルから作られます。
glyph outline と登録済み軸の値は現在のディスク上のファイルから遅延読み込みされます。
Dataset object の lifetime 中にフォントファイルを変更すること、pickle/unpickle
境界をまたいで変更することは unsupported で、sample と label の不整合を起こす
可能性があります。

## 参照型

```python
from torchfont.structures import (
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
| `GlyphSample` | `ref: GlyphRef`, `font_idx: int`, `style_idx: int`, `character_idx: int`, `weight: float \| None`, `width: float \| None`, `italic: float \| None`, `slant: float \| None`, `optical_size: float \| None` |
| `VariableGlyphSample` | `ref: VariableGlyphRef`, `font_idx: int`, `character_idx: int` |

`ttc_index` は read-fonts/skrifa が TrueType Collection 内のフォント位置に
使っている名前に合わせています。単一フォントのファイルでは `0` です。

## GlyphDataset

```python
from torchfont.datasets import GlyphDataset
from torchfont.instance_fn import named_instances

dataset = GlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),
    patterns=("**/*.ttf",),
    instance_fn=named_instances,
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
    instance_fn: InstanceLocationsFn = torchfont.instance_fn.named_instances,
    transform: Callable[[GlyphSample], T] | None = None,
)
```

targets:

- `font_targets -> LongTensor (N,)`
- `style_targets -> LongTensor (N,)`
- `character_targets -> LongTensor (N,)`
- `weight_targets -> FloatTensor (N,)`
- `width_targets -> FloatTensor (N,)`
- `italic_targets -> FloatTensor (N,)`
- `slant_targets -> FloatTensor (N,)`
- `optical_size_targets -> FloatTensor (N,)`

これらの target は OpenType の登録済み user scale を使うため、weight は CSS weight
と比較可能、width は百分率、italic は
`0`（Roman）から `1`（fully italic）、slant は度、optical size は point です。
`italic_targets` の中間 variation 座標を含め、5つの target はすべて浮動小数点数です。
各軸についてまず index 済み variation location を使い、`fvar` に存在しない軸だけを
対応する OS/2、`head`、`post` field から変換します。fallback は `wght` に OS/2
`usWeightClass`、`wdth` に
OS/2 `usWidthClass`、`ital` に OS/2 `fsSelection.ITALIC`（OS/2 がない場合は
`head.macStyle.ITALIC`）、`slnt` に `post.italicAngle` を使います。
`head.macStyle.BOLD` は weight class ではないため、恣意的な `wght` 値には変換しません。
フォントから導出できない値は `NaN` になるため、loss から除外する場合は
`torch.isfinite` をそのまま利用できます。
同じ値は各 sample の `sample.weight`、`sample.width`、`sample.italic`、
`sample.slant`、`sample.optical_size` からも取得できるため、transform 内で利用できます。
sample で取得できない値は `None`（target Tensor では `NaN`）です。
これらの target property にアクセスした時点でフォントファイルを parse します。
展開済みtarget vectorはDataset構築時には作られず、cacheもされません。
OS/2 の optical-size range は恣意的な中点に変換しません。`opsz` は index 済み
variation location が実際の座標を持つ場合だけ値を持ちます。

class 語彙:

- `font_classes -> list[FontRef]`
- `style_classes -> list[str]`
- `character_classes -> list[str]`
- `character_class_to_idx -> dict[str, int]`

## VariableGlyphDataset

```python
from torchfont.datasets import VariableGlyphDataset
from torchfont.instance_fn import named_instance_count

dataset = VariableGlyphDataset(
    root="~/fonts",
    codepoints=range(0x41, 0x5B),
    instance_fn=named_instance_count,
)
```

`VariableGlyphDataset` は location を index に含めません。各アクセスで transform が
新しい location をサンプルする training augmentation に向いています。instance-count function
は各フォントの離散的な多重度を決めます。静的フォントも通常の
フォントとして含まれます。

コンストラクタ:

```python
VariableGlyphDataset(
    root: Path | str,
    *,
    instance_fn: InstanceCountFn = torchfont.instance_fn.named_instance_count,
    codepoints: Sequence[SupportsIndex] | None = None,
    patterns: Sequence[str] | None = None,
    transform: Callable[[VariableGlyphSample], T] | None = None,
)
```

targets:

- `font_targets -> LongTensor (N,)`
- `character_targets -> LongTensor (N,)`

## Instance Functions

```python
from torchfont.instance_fn import (
    default_instance,
    default_instance_count,
    grid_instance_count,
    grid_instances,
    named_instance_count,
    named_instances,
)
```

- `named_instances(font)`: fvar named instance を dedupe し、なければ default
- `default_instance(font)`: default location 1 つ
- `grid_instances({"wght": 7, "wdth": 3})`: 等間隔の固定 grid
- `grid_instances({})`: default location 1つを返す空gridのidentity
- `named_instance_count(font)`: `named_instances` に対応する個数
- `default_instance_count(font)`: instance slot 1 つ
- `grid_instance_count({"wght": 7, "wdth": 3})`: `grid_instances` に対応する個数

grid policyは指定していない軸をdefaultに固定し、個別fontが持たない指定軸を無視します。
このため一つのpolicyを異種font collectionへ適用できます。

transform 時の variation sampling には [Transform Utilities](./transforms.md) の
`RandomLocation` を使います。dataset-level seed はありません。

カスタム instance function は 0 個の location を返せます。カスタム関数が返した
未知の軸や、正規化後に重複する location は Dataset 構築時に `ValueError` になります。
