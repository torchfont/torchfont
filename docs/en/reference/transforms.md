# Transform API

`torchfont.transforms` provides composable preprocessing utilities with the
signature `GlyphSample -> GlyphSample`.

Migration note: the old tuple contract `(types, coords) -> (types, coords)` was removed.

## Compose

```python
from torchfont.transforms import Compose
```

```python
Compose(
    transforms: Sequence[Callable[[GlyphSample], GlyphSample]],
)
```

Applies transforms in order.

### Example (`Compose`)

```python
from torchfont.transforms import Compose, LimitSequenceLength, Patchify, QuadToCubic

transform = Compose([
    QuadToCubic(),
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])
```

---

## LimitSequenceLength

```python
from torchfont.transforms import LimitSequenceLength
```

```python
LimitSequenceLength(max_len: int)
```

Returns a `GlyphSample` whose `types` and `coords` are truncated to `max_len`.

- no padding
- elements beyond `max_len` are truncated
- `max_len` must be `>= 0`; invalid values raise `ValueError` at construction

### I/O shape (`LimitSequenceLength`)

- input: `types=(seq_len,)`, `coords=(seq_len, 6)`
- output: `types=(min(seq_len, max_len),)`, `coords=(min(seq_len, max_len), 6)`

### Notes

- malformed input samples also raise `ValueError`; `LimitSequenceLength`
  expects untruncated tensors with `types.ndim == 1`, `coords.ndim == 2`,
  `coords.shape[1] == 6`, and aligned leading sequence lengths

---

## QuadToCubic

```python
from torchfont.transforms import QuadToCubic
```

```python
QuadToCubic()
```

Converts `CommandType.QUAD_TO` commands to `CommandType.CURVE_TO`.

- command shape is unchanged
- coordinate shape is unchanged (`(..., 6)`)
- for each quadratic segment, `[cx0, cy0, 0, 0, x, y]` is rewritten to cubic
  control points using the previous endpoint

### I/O shape (`QuadToCubic`)

- input: `types=(...)`, `coords=(..., 6)`
- output: `types=(...)`, `coords=(..., 6)`

---

## Patchify

```python
from torchfont.transforms import Patchify
```

```python
Patchify(patch_size: int)
```

Pads the sequence tail with zeros to align length to `patch_size`, then reshapes
into patches.

### I/O shape (`Patchify`)

- input: `types=(seq_len,)`, `coords=(seq_len, 6)`
- output: `types=(num_patches, patch_size)`,
  `coords=(num_patches, patch_size, 6)`
- `num_patches = ceil(seq_len / patch_size)`

### Notes

- type padding value: `0` (`pad`)
- coordinate padding value: `0.0`
- `patch_size` must be `>= 1`; invalid values raise `ValueError` at construction
- malformed input samples also raise `ValueError`; `Patchify` expects
  unpatchified tensors with `types.ndim == 1`, `coords.ndim == 2`,
  `coords.shape[1] == 6`, and aligned leading sequence lengths

### Example (`Patchify`)

```python
patchify = Patchify(patch_size=32)
patch_sample = patchify(sample)
```
