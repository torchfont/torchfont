# Semantic Structures

## `torchfont.structures`

Immutable semantic records and shared outline-encoding types. These values carry
font meaning through datasets and transform pipelines without adding tensor
subclass behavior.

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

`VariationLocation` is an immutable, hashable mapping. Axis tags are stored in
deterministic sorted order. `FontRef`, glyph references, and dataset samples are
frozen dataclasses and remain pickle-friendly for multiprocessing data loaders.

`Outline` couples the element-type and coordinate tensors. `GlyphData` couples a
transformed payload with its source sample and resolved variation location.

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
