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
    VariableGlyphRef,
    VariableGlyphSample,
)
```

`VariationLocation` は変更不可で hash 可能な mapping で、axis tag を常に
ソート済みの順序で保持します。`FontRef`、glyph reference、dataset sample は
frozen dataclass で、multiprocessing data loader でも pickle 可能です。

`Outline` は element-type tensor と coordinate tensor をひとまとまりにします。
`GlyphData` は transform 後の payload、元の sample、確定済み variation location
をひとまとまりにします。

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
