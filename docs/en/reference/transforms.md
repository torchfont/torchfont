# Transform Utilities

`torchfont.transforms` provides small utility functions for adapting glyph
tensors. Keep dataset item shaping in your own preprocessing code.

## quad_to_cubic

```python
from torchfont.transforms import quad_to_cubic
```

```python
types, coords = quad_to_cubic(types, coords)
```

Converts `CommandType.QUAD_TO` commands to `CommandType.CURVE_TO`.

- command shape is unchanged
- coordinate shape is unchanged (`(..., 6)`)
- for each quadratic segment, `[cx0, cy0, 0, 0, x, y]` is rewritten to cubic
  control points using the previous endpoint
- the last `types` dimension is the sequence dimension; leading dimensions are
  treated as independent sequences

Call `quad_to_cubic` before chunking one continuous outline into patches when
endpoint continuity must cross chunk boundaries.

### I/O Shape

- input: `types=(...)`, `coords=(..., 6)`
- output: `types=(...)`, `coords=(..., 6)`
