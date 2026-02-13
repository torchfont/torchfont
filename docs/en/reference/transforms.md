# Transform API

`torchfont.transforms` provides composable preprocessing utilities with the
signature `(types, coords) -> (types, coords)`.

## Compose

```python
from torchfont.transforms import Compose
```

```python
Compose(
    transforms: Sequence[Callable[[Tensor, Tensor], tuple[Tensor, Tensor]]],
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

Returns `types[:max_len]` and `coords[:max_len]`.

- no padding
- elements beyond `max_len` are truncated

### I/O shape (`LimitSequenceLength`)

- input: `types=(seq_len,)`, `coords=(seq_len, d)`
- output: `types=(min(seq_len, max_len),)`, `coords=(min(seq_len, max_len), d)`

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

- input: `types=(seq_len,)`, `coords=(seq_len, d)`
- output: `types=(num_patches, patch_size)`,
  `coords=(num_patches, patch_size, d)`
- `num_patches = ceil(seq_len / patch_size)`

### Notes

- type padding value: `0` (`pad`)
- coordinate padding value: `0.0`
- `patch_size` must be `> 0` (`0` or negative values fail at runtime)

### Example (`Patchify`)

```python
patchify = Patchify(patch_size=32)
patch_types, patch_coords = patchify(types, coords)
```
