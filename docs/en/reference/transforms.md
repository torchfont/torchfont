# Transform Utilities

`torchfont.transforms` provides stateless utility functions over `types` and
`coords` tensors. They do not define a fixed pipeline abstraction; write the
`GlyphDataset(transform=...)` callable or training-step preprocessing that fits
your workload.

## `limit_sequence_length`

```python
from torchfont.transforms import limit_sequence_length

types, coords = limit_sequence_length(types, coords, max_len=512)
```

Returns `(types, coords)` truncated to `max_len`.
`max_len` must be `>= 0`.

## `quad_to_cubic`

```python
from torchfont.transforms import quad_to_cubic

types, coords = quad_to_cubic(types, coords)
```

Converts `CommandType.QUAD_TO` commands to `CommandType.CURVE_TO` while keeping
the command and coordinate shapes unchanged.

## `patchify`

```python
from torchfont.transforms import patchify

types, coords = patchify(types, coords, patch_size=32)
```

Pads the sequence tail with zeros to align length to `patch_size`, then reshapes
into patches:

- input: `types=(seq_len,)`, `coords=(seq_len, 6)`
- output: `types=(num_patches, patch_size)`,
  `coords=(num_patches, patch_size, 6)`

`patch_size` must be `>= 1`.

## Example

```python
import dataclasses

from torchfont.transforms import limit_sequence_length, patchify, quad_to_cubic


def transform(sample):
    types, coords = quad_to_cubic(sample.types, sample.coords)
    types, coords = limit_sequence_length(types, coords, max_len=512)
    types, coords = patchify(types, coords, patch_size=32)
    return dataclasses.replace(sample, types=types, coords=coords)
```
