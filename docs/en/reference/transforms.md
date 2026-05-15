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


## remove_overlaps

```python
from torchfont.transforms import remove_overlaps
```

```python
types, coords = remove_overlaps(types, coords)
```

Merges overlapping glyph contours with Skia PathOps while preserving winding-based holes.

- accepts one continuous outline sequence
- removes internal overlap edges and returns a new variable-length outline
- returns the original outline unchanged when Skia PathOps cannot simplify it
- output tensors are CPU-backed because PathOps runs in the Rust backend

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(M,)`, `coords=(M, 6)`

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
bitmap = render_bitmap(types, coords, size=64, mode="bbox_square")
```

Renders a glyph outline to a greyscale bitmap tensor. `mode` controls how
coordinates are mapped to the output bitmap.

- `size` must be between 1 and 4096 (default: 64)
- `mode="fixed"` maps the fixed UPM-normalised range `[-0.25, 1.25] x [-0.25, 1.25]`
- `mode="bbox"` keeps the fixed-mode scale and returns a variable-size bitmap
  cropped to the tight glyph bounding box
- `mode="bbox_square"` scales the tight glyph bounding box uniformly and centres
  it (default)
- rendering uses the clipped / pre-patchified outline for faithful shape

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `uint8` tensor with values in `[0, 255]`; shape is `(size, size)` for
  `fixed` / `bbox_square` and variable `(height, width)` for `bbox`
