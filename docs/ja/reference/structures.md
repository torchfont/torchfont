# 意味構造

## `torchfont.structures`

変更不可の意味レコードとアウトラインのエンコーディングに使う共通型です。
テンソルサブクラスの挙動を追加せず、データセットと変換パイプラインの間で
フォントの意味を保持します。

```python
from torchfont.structures import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
    VariationLocation,
)
```

`VariationLocation` は変更不可でハッシュ可能なマッピングで、軸タグを常に
ソート済みの順序で保持します。イテラブルな入力に正規化後の重複タグがある場合は、
値を黙って上書きせず拒否します。`FontRef`、グリフ参照、データセットのサンプルは
変更不可の `dataclass` で、マルチプロセスのデータローダーでも `pickle` 可能です。

`Outline` は一つの可変長で、バッチ化されていないグリフを表します。`types` は形状 `(N,)` の
`torch.long`、`coords` は形状 `(N, 6)` の `torch.float32` で、各行が一対一に対応し、
同じデバイス上に置かれます。これらの構造的不変条件は `Outline` 構築時に検査されます。
各要素型で使われない座標、および `CLOSE`、`END`、`PAD` の全座標には意味が
ありません。`GlyphData` は変換後の Payload、Glyph 参照、確定済みの Variation Location、
並列な `font_idx`、`character_idx`、`weight`、`width`、`italic`、`slant`、`optical_size`
Target を浅い一つの Record に保持します。Registered Axis の値を優先し、Axis にない値は
可能な場合に OS/2、`head`、`post` の同等な Metadata から取得します。取得できない Target
は `NaN` です。

`Outline` と `GlyphData` は意図的に同一性による比較を使います。Tensor Payload には
`dataclass` が生成する等値比較ではなく、`torch.equal()` などを使います。`GlyphSample` は
値 Record のままです。

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

### `TYPE_DIM: int`

要素型の数。現在値は `7`。

### `COORD_DIM: int`

座標の次元数。現在値は `6`（`[cx0, cy0, cx1, cy1, x, y]`）。
