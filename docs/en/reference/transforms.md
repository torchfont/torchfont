# Transform Utilities

`torchfont.transforms` provides small utility functions for adapting glyph
tensors. Keep dataset item shaping in your own preprocessing code.

## QuadToCubic

```python
from torchfont.transforms import QuadToCubic
```

```python
types, coords = QuadToCubic(types, coords)
```

Converts `CommandType.QUAD_TO` commands to `CommandType.CURVE_TO`.

- command shape is unchanged
- coordinate shape is unchanged (`(..., 6)`)
- for each quadratic segment, `[cx0, cy0, 0, 0, x, y]` is rewritten to cubic
  control points using the previous endpoint

### I/O Shape

- input: `types=(...)`, `coords=(..., 6)`
- output: `types=(...)`, `coords=(..., 6)`
