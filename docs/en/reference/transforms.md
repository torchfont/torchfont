# Transform Utilities

`torchfont.transforms` provides small utility functions for adapting glyph
samples and tensors. Keep dataset item shaping in your own preprocessing code.

## load_glyph

```python
from torchfont.transforms import load_glyph

types, coords = load_glyph(sample.ref)
```

`load_glyph` is the bridge from a dataset glyph reference to the `(types,
coords)` pair the rest of this module operates on — typically the first call
inside a `GlyphDataset`/`VariableGlyphDataset` `transform`. It returns
`(types, coords)`, where `types` is a 1-D `LongTensor` and `coords` is a 2-D
`FloatTensor` of shape `(N, 6)`.

Outline-to-outline transforms preserve their input devices. Native outline
operations run in Rust on CPU and move their results back to the corresponding
input devices before returning.

For `VariableGlyphRef`, pass a location explicitly:

```python
from torchfont.transforms import random_location

sample = variable_dataset[0]
location = random_location(sample.ref.font)
types, coords = load_glyph(sample.ref, location)
```

For explicit locations, unknown axes, out-of-range values, and NaN/inf values
raise `ValueError`. Missing axes use the font default.

## random_location

```python
from torchfont.transforms import random_location

location = random_location(sample.ref.font, generator=None)
```

Samples each variation axis independently and uniformly over its user-space
minimum and maximum. Static fonts return an empty dictionary. Randomness uses
the optional `torch.Generator`, including a CUDA generator when provided.

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

- path element count is unchanged
- for each quadratic segment, `[cx0, cy0, 0, 0, x, y]` is rewritten to cubic
  control points using the previous endpoint

Set `merge_curves=True` to merge adjacent reconstructable curves and collinear
lines in the same Rust call immediately after conversion. This mode is useful
after `cubic_to_quad`.

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)`, `coords=(N, 6)`
- with `merge_curves=True`: output `types=(M,)`, `coords=(M, 6)` (length may differ)

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

## normalize_subpath_start_points

```python
from torchfont.transforms import normalize_subpath_start_points
```

```python
types, coords = normalize_subpath_start_points(types, coords)
```

Moves each subpath start to its lexicographically smallest `(x, y)` endpoint.

- only closed subpaths are changed; open subpaths are left unchanged
- when rotation crosses the old closing edge, that implicit edge is materialised as `LineTo`
- the represented geometry is preserved; rotating away from the original start may add one `LineTo`

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(M,)`, `coords=(M, 6)`

## randomize_subpath_start_points

```python
from torchfont.transforms import randomize_subpath_start_points
```

```python
types, coords = randomize_subpath_start_points(types, coords)
```

Chooses a uniformly random start endpoint for each subpath.

- useful as augmentation when a model should not depend on subpath start-point position
- only closed subpaths are changed; open subpaths are left unchanged
- `generator`: optional `torch.Generator` for reproducibility

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(M,)`, `coords=(M, 6)`

## randomize_subpath_order

```python
from torchfont.transforms import randomize_subpath_order

types, coords = randomize_subpath_order(types, coords, generator=None)
```

Randomly permutes complete subpaths while preserving every subpath's geometry,
start point, winding, and open/closed state. This helps prevent sequence models
from learning an arbitrary contour order.

For an extracted monochrome outline, contour order does not determine holes:
the winding/fill rule does. This transform intentionally does not operate on
source-font point indices, TrueType instructions, variation deltas, composite
components, or color-glyph layers, whose ordering can carry other semantics.

## patchify

```python
from torchfont.transforms import patchify
```

```python
patch_types, patch_coords = patchify(types, coords, patch_size=32)
```

Pads a 1-D glyph sequence with zeros to the nearest multiple of `patch_size`,
then splits it into contiguous patches.

- padding is zero-filled and appended only when `seq_len % patch_size != 0`

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
- zero-coordinate element types (CLOSE, END, PAD) are not modified
- closed subpath winding is preserved by default
- pass `preserve_winding=False` to keep the reflected winding order instead
- open subpaths are reflected but not reversed

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)`, `coords=(N, 6)`

## vertical_flip

```python
from torchfont.transforms import vertical_flip
```

```python
types, coords = vertical_flip(types, coords)
```

Flips a glyph outline vertically around its tight bounding-box centre.

- both on-curve endpoints and off-curve control points are transformed
- zero-coordinate element types (CLOSE, END, PAD) are not modified
- closed subpath winding is preserved by default
- pass `preserve_winding=False` to keep the reflected winding order instead
- open subpaths are reflected but not reversed

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `types=(N,)`, `coords=(N, 6)`

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
transformed; zero-coordinate element types (CLOSE, END, PAD) are not modified.

- `angle`: counter-clockwise rotation in degrees (default: `0.0`)
- `translate`: translation `(tx, ty)` in em units; values must be finite (default: `(0.0, 0.0)`)
- `scale`: uniform scale factor, must be positive and finite (default: `1.0`)
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
- `preserve_winding`: preserve closed subpath winding after reflection (default: `True`)
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
- `preserve_winding`: preserve closed subpath winding after reflection (default: `True`)
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
- `translate`: maximum absolute translation `(max_dx, max_dy)` in em units;
  each axis is sampled from `[-max_d, max_d]` (default: no translation)
- `scale`: scale range `(min, max)`; both values must be positive and finite
  (default: no scaling)
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

- `std`: finite standard deviation in em units;
  `0.005` ≈ 5 font-units in a 1000-UPM font
- only active `coords` columns are perturbed (zero-coordinate element types
  CLOSE, END, PAD and unused zero-padding columns are left unchanged)
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
- `mode="fixed"` maps the fixed em-unit range `[-0.25, 1.25] x [-0.25, 1.25]`
- `mode="bbox"` keeps the fixed-mode scale and returns a variable-size bitmap
  cropped to the tight glyph bounding box
- `mode="bbox_square"` scales the tight glyph bounding box uniformly and centres
  it (default)
- rendering uses the clipped / pre-patchified outline for faithful shape

### I/O Shape

- input: `types=(N,)`, `coords=(N, 6)`
- output: `uint8` tensor with values in `[0, 255]`; shape is `(size, size)` for
  `fixed` / `bbox_square` and variable `(height, width)` for `bbox`
