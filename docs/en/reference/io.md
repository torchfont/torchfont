# IO API

## `torchfont.io`

Shared constants for glyph command encoding.

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

Number of command types. Current value: `7`.

### `COORD_DIM: int`

Coordinate dimension. Current value: `6` (`[cx0, cy0, cx1, cy1, x, y]`).
