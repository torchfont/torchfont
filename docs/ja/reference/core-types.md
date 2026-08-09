# 基本型

## `torchfont`

フォントデータをデータセットと変換パイプラインの間で受け渡すための型です。

```python
from torchfont import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
    pad_outlines,
    unpad_outlines,
)
```

Variation Location には `dict[str, float]` を使います。Font 参照、Glyph 参照、
Dataset Sample はマルチプロセスの DataLoader でも使用できます。

### `Outline`

`Outline` は結合した二つのテンソルを保持します。`types` は形状 `(*batch, N)` の
`torch.long`、`coords` は形状 `(*batch, N, 6)` の任意の浮動小数点型で、各行が
一対一に対応し、同じデバイス上に置かれます。各要素型で使われない座標、および
`CLOSE`、`END`、`PAD` の全座標には意味がありません。

単一グリフの Batch 形状は空です。[`pad_outlines`](#pad-outlines) は単一グリフを
バッチにまとめます。ほとんどの Transform は単一グリフだけを扱い、バッチ化された
`Outline` は明示的なエラーで拒否します。

次のプロパティで、中身を取り出さずに Outline を調べられます。

| プロパティ | 意味 |
| --- | --- |
| `shape` | `types` の形状、つまり `(*batch, N)` |
| `batch_shape` | 先頭の Batch 次元。単一グリフでは空 |
| `num_elements` | パディングを含む 1 グリフあたりの要素数 |
| `is_batched` | Batch 次元を持つかどうか |
| `dtype` | `coords` の浮動小数点型 |
| `device` | 両テンソルが共有するデバイス |
| `padding_mask` | 要素が `PAD` の位置で `True` になる Boolean Mask |

`to()` と `pin_memory()` は両方のテンソルに同時に適用されます。`to()` に渡す dtype は
`coords` にのみ適用され、浮動小数点型でなければなりません。それ以外の Tensor 操作は
`types` または `coords` に直接適用します。Index は座標軸を常に保ったまま論理的な `(*batch, N)` 次元を
対象にし、最後の要素次元を取り除く Index は使えません。`len()` は最初の論理次元の
長さを返します。`unbind()` は内容を変更せず、最初の
Batch 次元だけを分割します。

`Outline` 同士は同一性で比較されます。内容の等値性が必要な場合は `types` と
`coords` を `torch.equal()` で明示的に比較します。入力テンソルをその場で変更すると、
その変更は Outline にも反映されます。どちらの属性への代入もエラーになります。

### `pad_outlines`

```python
pad_outlines(outlines: Sequence[Outline]) -> Outline
```

単一 Outline を `ElementType.PAD` とゼロ座標でパディングして一つのバッチに
まとめます。入力はすべて単一グリフで、デバイスと `coords` の dtype が共通で
なければなりません。パディング位置は `padding_mask` で取得でき、
[`unpad_outlines`](#unpad-outlines) で元に戻せます。

### `unpad_outlines`

```python
unpad_outlines(outline: Outline) -> tuple[Outline, ...]
```

Batch 次元がちょうど一つの `Outline` を分割し、各結果の末尾にある
`ElementType.PAD` 行を除去します。これは `pad_outlines` の明示的な逆操作です。
Padding を保持する場合は `Outline.unbind()` を使います。

### `GlyphData`

`GlyphData` は変換後の Payload、Glyph 参照、Variation Location、
並列な `font_idx`、`character_idx`、`weight`、`width`、`italic`、`slant`、
`optical_size` Target を保持します。取得できない連続 Target は `None` です。

`location` は Glyph の読み込みに使った OpenType Axis Tag と値を保持する通常の
`dict[str, float]` です。

Index は Python の整数、連続 Target は Float です。取得できない連続 Target は
`None` になります。

`GlyphData` 同士は同一性で比較されます。Tensor Payload の内容を比較する場合は、
明示的な Tensor 比較を使ってください。

モデルの入力契約に合わせてローカルな `DataLoader.collate_fn` を定義し、Payload が可変長 Outline の
場合は [`pad_outlines`](#pad-outlines) を使います。
[DataLoader](../guide/dataloader.md) を参照してください。

### `ElementType: IntEnum`

```python
class ElementType(IntEnum):
    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6
```

`PAD` はフォントの読み込みでは生成されません。バッチのパディングで導入された行を
標識します。

### `TYPE_DIM: int`

要素型の数。現在値は `7`。

### `COORD_DIM: int`

座標の次元数。現在値は `6`（`[cx0, cy0, cx1, cy1, x, y]`）。
