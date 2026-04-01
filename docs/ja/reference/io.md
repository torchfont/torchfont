# IO API

## `torchfont.io`

グリフコマンドの共通定数です。

```python
from torchfont.io import CommandType, TYPE_DIM, COORD_DIM
```

### `CommandType: IntEnum`

```python
class CommandType(IntEnum):
    PAD = 0
    MOVE_TO = 1
    LINE_TO = 2
    QUAD_TO = 3
    CURVE_TO = 4
    CLOSE = 5
    END = 6
```

### `TYPE_DIM: int`

コマンド種別数。現在値は `7`。

### `COORD_DIM: int`

座標次元数。現在値は `6`（`[cx0, cy0, cx1, cy1, x, y]`）。
