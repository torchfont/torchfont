# Core Types

## `torchfont`

Immutable semantic records and shared outline-encoding types. These values carry
font meaning through datasets and transform pipelines without adding tensor
subclass behavior.

```python
from torchfont import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphRef,
    GlyphSample,
    Outline,
    VariationLocation,
)
```

`VariationLocation` is an immutable, hashable mapping. Axis tags are stored in
deterministic sorted order, and iterable inputs with duplicate normalized tags
are rejected rather than silently overwriting a value. `FontRef`, glyph
references, and dataset samples are frozen dataclasses and remain pickle-friendly
for multiprocessing data loaders.

`Outline` represents one variable-length, unbatched glyph. `types` has shape
`(N,)`, `coords` has shape `(N, 6)`, and their rows correspond one-to-one.
`types` uses `torch.long`, `coords` uses `torch.float32`, and both tensors are
on the same device. These structural invariants are checked when an `Outline`
is constructed. Coordinates that are inactive for an element type, including
all coordinates for `CLOSE`, `END`, and `PAD`, have no semantic value.
`GlyphData` is a shallow record containing a transformed payload, glyph
reference, resolved variation location, and the parallel `font_idx`,
`character_idx`, `weight`, `width`, `italic`, `slant`, and `optical_size`
targets. Registered variation-axis values take precedence; values not
represented by an axis are read from equivalent OS/2, `head`, and `post`
metadata when available. Unavailable targets are `NaN`.

`Outline` and `GlyphData` intentionally use identity equality. Their tensor
payloads require explicit comparisons such as `torch.equal()` rather than
dataclass-generated equality. `GlyphSample` remains a value record.

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

Number of element types. Current value: `7`.

### `COORD_DIM: int`

Coordinates width. Current value: `6` (`[cx0, cy0, cx1, cy1, x, y]`).
