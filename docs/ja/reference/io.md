# IO API

## `torchfont.io`

グリフ outline の path element encoding に使う共通定数です。

```python
from torchfont.io import ElementType, TYPE_DIM, COORD_DIM
```

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
