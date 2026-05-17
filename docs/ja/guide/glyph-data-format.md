# グリフデータ形式

<!-- markdownlint-disable MD013 -->

TorchFont の Dataset は、各グリフを `GlyphSample` として返します。

```python
sample = dataset[i]
```

| フィールド | 型 | 形状 | 意味 |
| --- | --- | --- | --- |
| `sample.types` | `torch.LongTensor` | `(seq_len,)` | element type 列 |
| `sample.coords` | `torch.FloatTensor` | `(seq_len, 6)` | 各 path element の coordinates |
| `sample.style_idx` | `int` | スカラー | 書体スタイルのクラスID |
| `sample.content_idx` | `int` | スカラー | 文字内容のクラスID |
| `sample.metrics` | `torch.FloatTensor` | `(15,)` | グリフ・フォント単位のメトリクス |
| `sample.glyph_name` | `str` | — | PostScript グリフ名 |

`GlyphSample` は dataclass なので、名前付きフィールドでアクセスする使い方が前提です。

## Outline モデル

- **Outline**: path element の系列
- **Subpath**: `MoveTo` ではじまり `Close` で終わる path element の系列（`Close` がない open subpath も便宜的に subpath として扱う）
- **Path element**: 1 つの element type と 1 行の coordinates
- **Element type**: `MoveTo`, `LineTo`, `QuadTo`, `CurveTo`, `Close`, `End`, `Pad`
- **Coordinates**: `[cx0, cy0, cx1, cy1, x, y]`

`sample.metrics` は形状 `(15,)` の 1-D `float32` テンソルです:

```python
# [adv_w, lsb, x_min, y_min, x_max, y_max,
#  ascent, descent, leading, cap_height, x_height, avg_width,
#  italic_angle, units_per_em, is_monospace]
m = sample.metrics
```

値は UPM で正規化済み（該当する場合）。欠損時は `nan`。

## `types` の定義

```python
from torchfont.io import ElementType

print(ElementType.QUAD_TO, ElementType.QUAD_TO.value)
# ElementType.QUAD_TO 3
```

- `ElementType.END` はシーケンス終端
- `ElementType.PAD` は主に `pad_sequence` や独自 padding で出現

## `coords` の定義

各ステップは 6 次元です。

```text
[cx0, cy0, cx1, cy1, x, y]
```

- `ElementType.MOVE_TO` / `ElementType.LINE_TO`: 制御点は 0、終点 `(x, y)` を使用
- `ElementType.QUAD_TO`: 2 次ベジェの制御点 `(cx0, cy0)` と終点 `(x, y)` を使用（`cx1`,
  `cy1` は 0）
- `ElementType.CURVE_TO`: 3 次ベジェの制御点 `(cx0, cy0)` と `(cx1, cy1)` + 終点を使用
- `ElementType.CLOSE` / `ElementType.END` / `ElementType.PAD`: coordinates は 0

::: info
coordinates はフォントの `units_per_em` で正規化されています。
:::

## 2 次ベジェの扱い

2 次ベジェは `ElementType.QUAD_TO` としてそのまま出力されます。テンソル形状を固定するため、`ElementType.QUAD_TO` の coordinates は `[cx0, cy0, 0, 0, x, y]` です。

## スタイルとコンテンツのラベル

### `style_idx`

- 静的フォント: 通常は `Family + Subfamily` 形式（例: `Lato Regular`）
- 可変フォント: 名前付きインスタンスがあれば、その単位で 1 クラス
- 可変フォントでインスタンス名が空の場合: `Family` のみ
- 可変フォントで名前付きインスタンスがない場合: `Family + Subfamily`（または
  `Family`）へフォールバック

```python
metadata = dataset.metadata

print(dataset.style_classes[:5])
print(metadata.styles[:5])
print(metadata.style_name_to_idxs)
```

`metadata.styles` は relative path / face / instance 由来の衝突しない識別子を持ち、重複表示名は `metadata.style_name_to_idxs` で全 index を取得できます。

### `content_idx`

- Unicode 文字ごとのクラスID

```python
metadata = dataset.metadata

print(dataset.content_classes[:5])
print(metadata.contents[:5])
print(metadata.content_id_to_idx)
```

## `targets` で一括取得

```python
t = dataset.targets  # shape: (N, 2), dtype: torch.long

style_all = t[:, 0]    # 1 列目: style_idx
content_all = t[:, 1]  # 2 列目: content_idx
```

## Utility 後の形状

`quad_to_cubic` は `types` / `coords` の形状を保持します。

`patchify` は形状を変更します。長さ `N` のシーケンスは `types` が
`(num_patches, patch_size)`、`coords` が `(num_patches, patch_size, 6)` になります。
モデル固有の整形を dataset transform として追加する場合も、返す sample の
`style_idx` / `content_idx` との対応を保ってください。
