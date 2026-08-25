# Core Types

## `torchfont`

Types for carrying font data through datasets and transform pipelines.

```python
from torchfont import (
    CodepointData,
    CodepointSample,
    ElementType,
    FontRef,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
    Outline,
)
```

Variation locations use `dict[str, float]` values. Font references, glyph
references, and dataset samples can be used with multiprocessing data loaders.

### `Outline`

`Outline` pairs two coupled tensors. `types` has shape `(N,)`,
`coords` has shape `(N, 6)`, and their rows correspond one-to-one. `types` uses
`torch.long`, `coords` uses any floating point dtype, and both tensors are on the
same device. Coordinates that are inactive for an element type,
including all coordinates for `CLOSE`, `END`, and `PAD`, have no semantic value.

These properties describe an outline without taking it apart:

| Property | Meaning |
| --- | --- |
| `shape` | shape of `types`, that is `(N,)` |
| `num_elements` | number of path elements |
| `dtype` | floating point dtype of `coords` |
| `device` | device shared by both tensors |

`to()` and `pin_memory()` apply to both tensors at once. A dtype passed to
`to()` applies to `coords` only and must be floating point. Operate on `types`
or `coords` directly for other tensor operations. Indexing preserves the
element dimension and coordinate axis. `len()` reports the number of elements.

`Outline` objects compare by identity. Compare `types` and `coords` explicitly
with `torch.equal()` when content equality is required. In-place changes to the
input tensors are visible through the outline. Assigning a different tensor to
either attribute raises an error.

### `CodepointData`

`CodepointData` contains a transformed payload, glyph reference, resolved
variation location, and the parallel `codepoint`,
`font_idx`, `character_idx`, `weight`, `width`, `italic`, `slant`, and
`optical_size` targets. Unavailable continuous targets are `None`.

`location` is an ordinary `dict[str, float]` mapping OpenType axis tags to the
values used to load the glyph.

Indices are Python integers and continuous targets are floats. An unavailable
continuous target is `None`.

`CodepointData` objects compare by identity. Compare tensor payloads explicitly
when content equality is required.

Define a local `DataLoader.collate_fn` for the model's input contract and use
PyTorch's `pad_sequence` when its payloads are variable-length outlines. See
[DataLoader](../guide/basic/dataloader.md).

### `GlyphIdData`

`GlyphIdData` is what `LoadGlyph` returns for a `GlyphIdSample`. It carries the
same payload, glyph reference, resolved variation location, and `font_idx`,
`weight`, `width`, `italic`, `slant`, and `optical_size` targets as
`CodepointData`, without the `codepoint` and `character_idx` targets that a
glyph no character maps to cannot have.

It follows the same rules as `CodepointData`: identity equality, integer
indices, and `None` for an unavailable continuous target.

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

`PAD` is never produced by loading a font. It marks rows introduced by padding a
batch.
