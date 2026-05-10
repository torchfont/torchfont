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

## patchify

```python
from torchfont.transforms import patchify
```

```python
patch_types, patch_coords = patchify(types, coords, patch_size=32)
```

Pads a 1-D glyph sequence with zeros to the nearest multiple of `patch_size`,
then splits it into contiguous patches.

- `patch_size` must be >= 1
- padding is zero-filled and appended only when `seq_len % patch_size != 0`
- call `quad_to_cubic` before `patchify` when endpoint continuity must cross
  patch boundaries

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(num_patches, patch_size)`, `coords=(num_patches, patch_size, 6)`

## render_bitmap

```python
from torchfont.transforms import render_bitmap
```

```python
bitmap = render_bitmap(types, coords, size=64)
```

Renders a glyph outline to a greyscale bitmap tensor. The glyph is auto-scaled
and centred to fill the canvas with a fixed 4-pixel padding on each side.

- `size` must be between 1 and 4096 (default: 64)
- rendering uses the clipped / pre-patchified outline for faithful shape

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `uint8` tensor of shape `(size, size)` with values in `[0, 255]`
