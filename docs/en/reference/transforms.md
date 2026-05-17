# Transform Utilities

`torchfont.transforms` provides small utility functions for adapting glyph
tensors. Keep dataset item shaping in your own preprocessing code.

## quad_to_cubic

```python
from torchfont.transforms import quad_to_cubic
```

```python
types, coords = quad_to_cubic(types, coords)
# or, for one continuous outline sequence:
types, coords = quad_to_cubic(types, coords, merge_curves=True)
```

Converts `ElementType.QUAD_TO` path elements to `ElementType.CURVE_TO`.

- path element shape is unchanged
- `coords` shape is unchanged (`(..., 6)`)
- for each quadratic segment, `[cx0, cy0, 0, 0, x, y]` is rewritten to cubic
  control points using the previous endpoint
- the last `types` dimension is the sequence dimension; leading dimensions are
  treated as independent sequences

Call `quad_to_cubic` before chunking one continuous outline into patches when
endpoint continuity must cross chunk boundaries.

Set `merge_curves=True` to merge adjacent reconstructable curves and collinear
lines in the same Rust call immediately after conversion. This mode is useful
after `cubic_to_quad`, and because merging can shorten the outline it accepts
one continuous outline sequence rather than batched inputs.

### I/O Shape

- input: `types=(...)`, `coords=(..., 6)`
- output: `types=(...)`, `coords=(..., 6)`
- with `merge_curves=True`: input `types=(N,)`, `coords=(N, 6)`; output
  `types=(M,)`, `coords=(M, 6)`

## cubic_to_quad

```python
from torchfont.transforms import cubic_to_quad
```

```python
types, coords = cubic_to_quad(types, coords)
```

Converts `ElementType.CURVE_TO` path elements to quadratic splines using the same
approximation strategy as fonttools cu2qu.

- accepts one continuous outline sequence
- one cubic may expand into multiple `ElementType.QUAD_TO` path elements
- adjacent quadratic controls imply on-curve points at their midpoints

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(M,)`, `coords=(M, 6)`

## merge_curves

```python
from torchfont.transforms import merge_curves
```

```python
types, coords = merge_curves(types, coords)
```

Collapses adjacent segments when they can be reconstructed as one parent shape.

- split cubic pieces are merged back into one cubic when the reconstruction fits
- split quadratic pieces are merged back into one quadratic when the reconstruction fits
- consecutive collinear `LineTo` path elements moving in the same direction are merged
- subpath boundaries are preserved

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(M,)`, `coords=(M, 6)`


## remove_overlaps

```python
from torchfont.transforms import remove_overlaps
```

```python
types, coords = remove_overlaps(types, coords)
```

Merges overlapping glyph subpaths with Skia PathOps while preserving winding-based holes.

- accepts one continuous outline sequence
- removes internal overlap edges and returns a new variable-length outline
- returns the original outline unchanged when Skia PathOps cannot simplify it

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

## horizontal_flip

```python
from torchfont.transforms import horizontal_flip
```

```python
types, coords = horizontal_flip(types, coords)
```

Flips a glyph outline horizontally around its tight bounding-box centre.

- both on-curve endpoints and off-curve control points are transformed
- padding entries (CLOSE, END, PAD) are not modified
- flipping reverses subpath winding order

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)` (unchanged), `coords=(N, 6)`

## vertical_flip

```python
from torchfont.transforms import vertical_flip
```

```python
types, coords = vertical_flip(types, coords)
```

Flips a glyph outline vertically around its tight bounding-box centre.

- both on-curve endpoints and off-curve control points are transformed
- padding entries (CLOSE, END, PAD) are not modified
- flipping reverses subpath winding order

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)` (unchanged), `coords=(N, 6)`

## affine

```python
from torchfont.transforms import affine
```

```python
types, coords = affine(types, coords, angle=15.0, translate=(0.05, 0.0), scale=0.9, shear=5.0)
```

Applies a deterministic affine transformation to a glyph outline.

Composes uniform scale, x-shear, and rotation around the tight bounding-box
centre, then applies `translate`. All active control points and endpoints are
transformed; padding entries are not modified.

- `angle`: counter-clockwise rotation in degrees (default: `0.0`)
- `translate`: translation `(tx, ty)` in UPM-normalised units (default: `(0.0, 0.0)`)
- `scale`: uniform scale factor, must be positive (default: `1.0`)
- `shear`: x-shear angle in degrees (default: `0.0`)

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)` (unchanged), `coords=(N, 6)`

## random_horizontal_flip

```python
from torchfont.transforms import random_horizontal_flip
```

```python
types, coords = random_horizontal_flip(types, coords, p=0.5)
```

Randomly applies `horizontal_flip` with probability `p`.

- `p`: flip probability (default: `0.5`)
- `generator`: optional `torch.Generator` for reproducibility

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)`, `coords=(N, 6)`

## random_vertical_flip

```python
from torchfont.transforms import random_vertical_flip
```

```python
types, coords = random_vertical_flip(types, coords, p=0.5)
```

Randomly applies `vertical_flip` with probability `p`.

- `p`: flip probability (default: `0.5`)
- `generator`: optional `torch.Generator` for reproducibility

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)`, `coords=(N, 6)`

## random_affine

```python
from torchfont.transforms import random_affine
```

```python
types, coords = random_affine(
    types, coords,
    degrees=15.0,
    translate=(0.05, 0.05),
    scale=(0.9, 1.1),
    shear=5.0,
)
```

Applies a random affine transformation sampled uniformly from the given ranges.

- `degrees`: rotation range in degrees; a single float `d` gives `[-d, d]`
- `translate`: maximum absolute translation `(max_dx, max_dy)` in UPM-normalised
  units; each axis is sampled from `[-max_d, max_d]` (default: no translation)
- `scale`: scale range `(min, max)`; both values must be positive (default: no scaling)
- `shear`: x-shear range in degrees; same format as `degrees` (default: `0.0`)
- `generator`: optional `torch.Generator` for reproducibility

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)` (unchanged), `coords=(N, 6)`

## random_coord_jitter

```python
from torchfont.transforms import random_coord_jitter
```

```python
types, coords = random_coord_jitter(types, coords, std=0.005)
```

Adds independent Gaussian noise to each active value in the outline coordinates.

- `std`: standard deviation in UPM-normalised units; `0.005` ≈ 5 font-units in
  a 1000-UPM font
- only active `coords` columns are perturbed (unused columns, CLOSE, END, PAD
  are left unchanged)
- `generator`: optional `torch.Generator` for reproducibility

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)` (unchanged), `coords=(N, 6)`

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
