# 基本型

## `torchfont`

フォントデータをデータセットと変換パイプラインの間で受け渡すための型です。

```python
from torchfont import (
    CodepointData,
    CodepointSample,
    ElementType,
    FontRef,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
    Outline,
)
```

Variation Location には `dict[str, float]` を使います。Font 参照、Glyph 参照、
Dataset Sample はマルチプロセスの DataLoader でも使用できます。

### `Outline`

`Outline` は結合した二つのテンソルを保持します。`types` は形状 `(N,)` の
`torch.long`、`coords` は形状 `(N, 6)` の任意の浮動小数点型で、各行が
一対一に対応し、同じデバイス上に置かれます。各要素型で使われない座標、および
`CLOSE`、`END`、`PAD` の全座標には意味がありません。

次のプロパティで、中身を取り出さずに Outline を調べられます。

| プロパティ | 意味 |
| --- | --- |
| `shape` | `types` の形状、つまり `(N,)` |
| `num_elements` | Path 要素数 |
| `dtype` | `coords` の浮動小数点型 |
| `device` | 両テンソルが共有するデバイス |

`to()` と `pin_memory()` は両方のテンソルに同時に適用されます。`to()` に渡す dtype は
`coords` にのみ適用され、浮動小数点型でなければなりません。それ以外の Tensor 操作は
`types` または `coords` に直接適用します。Index は要素次元と座標軸を保持します。
`len()` は要素数を返します。

`Outline` 同士は同一性で比較されます。内容の等値性が必要な場合は `types` と
`coords` を `torch.equal()` で明示的に比較します。入力テンソルをその場で変更すると、
その変更は Outline にも反映されます。どちらの属性への代入もエラーになります。

### `CodepointData`

`CodepointData` は変換後の Payload、Glyph 参照、Variation Location、
並列な `codepoint`、`font_idx`、`character_idx`、`weight`、`width`、`italic`、
`slant`、`optical_size` Target を保持します。取得できない連続 Target は `None` です。

`location` は Glyph の読み込みに使った OpenType Axis Tag と値を保持する通常の
`dict[str, float]` です。

Index は Python の整数、連続 Target は Float です。取得できない連続 Target は
`None` になります。

`CodepointData` 同士は同一性で比較されます。Tensor Payload の内容を比較する場合は、
明示的な Tensor 比較を使ってください。

モデルの入力契約に合わせてローカルな `DataLoader.collate_fn` を定義し、Payload が可変長 Outline の
場合は PyTorch の `pad_sequence` を使います。
[DataLoader](../guide/basic/dataloader.md) を参照してください。

### `GlyphIdData`

`GlyphIdData` は `GlyphIdSample` に `LoadGlyph` を適用した結果です。`CodepointData` と同じ
Payload、Glyph 参照、Variation Location、`font_idx`、`weight`、`width`、`italic`、`slant`、
`optical_size` Target を保持します。どの文字も対応しないグリフが持てない `codepoint` と
`character_idx` の Target はありません。

同一性による比較、Python の整数 Index、取得できない連続 Target が `None` になる点は
`CodepointData` と同じです。

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
