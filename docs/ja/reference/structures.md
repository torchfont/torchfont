# 意味構造

## `torchfont.structures`

変更不可の意味レコードと outline encoding に使う共通型です。
Tensor subclass の挙動を追加せず、dataset と transform pipeline の間で
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

`VariationLocation` は変更不可で hash 可能な mapping で、axis tag を常に
ソート済みの順序で保持します。iterable input に正規化後の重複tagがある場合は、
値を黙って上書きせず拒否します。`FontRef`、glyph reference、dataset sample は
frozen dataclass で、multiprocessing data loader でも pickle 可能です。

`Outline` は一つの可変長、非 batch glyph を表します。`types` はshape `(N,)` の
`torch.long`、`coords` はshape `(N, 6)` の`torch.float32`で、各行が一対一に対応し、
同じdevice上に置かれます。これらの構造的不変条件は`Outline`構築時に検査されます。
各element typeで使われない座標、および`CLOSE`、`END`、`PAD`の全座標には意味が
ありません。`GlyphData` はtransform後のpayload、元のsample、確定済みvariation
locationをひとまとまりにします。

`Outline`と`GlyphData`は意図的にidentity equalityを使います。tensor payloadの比較には
dataclassが生成するbool equalityではなく、`torch.equal()`などを明示的に使います。

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

element type 数。現在値は `7`。

### `COORD_DIM: int`

coordinates の次元数。現在値は `6`（`[cx0, cy0, cx1, cy1, x, y]`）。
