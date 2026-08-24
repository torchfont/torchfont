# Core Types

## `torchfont`

Types for carrying font data through datasets and transform pipelines.

```python
from torchfont import (
    ElementType,
    FontRef,
    GlyphData,
    GlyphIdData,
    GlyphIdSample,
    GlyphRef,
    GlyphSample,
    Outline,
    pad_outlines,
    unpad_outlines,
)
```

Variation locations use `dict[str, float]` values. Font references, glyph
references, and dataset samples can be used with multiprocessing data loaders.

### `Outline`

`Outline` pairs two coupled tensors. `types` has shape `(*batch, N)`, `coords`
has shape `(*batch, N, 6)`, and their rows correspond one-to-one. `types` uses
`torch.long`, `coords` uses any floating point dtype, and both tensors are on the
same device. Coordinates that are inactive for an element type,
including all coordinates for `CLOSE`, `END`, and `PAD`, have no semantic value.

A single glyph has an empty batch shape. [`pad_outlines`](#pad-outlines) stacks
single glyphs into a batch. Most transforms operate on single glyphs and reject a
batched `Outline` with an explicit error.

These properties describe an outline without taking it apart:

| Property | Meaning |
| --- | --- |
| `shape` | shape of `types`, that is `(*batch, N)` |
| `batch_shape` | leading batch dimensions, empty for a single glyph |
| `num_elements` | path elements per glyph, including padding |
| `is_batched` | whether any batch dimension is present |
| `dtype` | floating point dtype of `coords` |
| `device` | device shared by both tensors |
| `padding_mask` | boolean mask, `True` where an element is `PAD` |

`to()` and `pin_memory()` apply to both tensors at once. A dtype passed to
`to()` applies to `coords` only and must be floating point. Operate on `types`
or `coords` directly for other tensor operations. Indexing addresses the
logical `(*batch, N)` dimensions while
always leaving the coordinate axis intact. An index may not remove the final
element dimension. `len()` reports the first logical dimension. `unbind()`
splits the first batch dimension without changing its contents.

`Outline` objects compare by identity. Compare `types` and `coords` explicitly
with `torch.equal()` when content equality is required. In-place changes to the
input tensors are visible through the outline. Assigning a different tensor to
either attribute raises an error.

### `pad_outlines`

```python
pad_outlines(outlines: Sequence[Outline]) -> Outline
```

Stacks single outlines into one batch padded with `ElementType.PAD` and zero
coordinates. Every input must be a single glyph sharing one device and one
`coords` dtype. Recover the padding with `padding_mask`, and undo it with
[`unpad_outlines`](#unpad-outlines).

### `unpad_outlines`

```python
unpad_outlines(outline: Outline) -> tuple[Outline, ...]
```

Splits an `Outline` with exactly one batch dimension and removes trailing
`ElementType.PAD` rows from each result. This is the explicit inverse of
`pad_outlines`; use `Outline.unbind()` when padding must be preserved.

### `GlyphData`

`GlyphData` contains a transformed payload, glyph
reference, resolved variation location, and the parallel `codepoint`,
`font_idx`, `character_idx`, `weight`, `width`, `italic`, `slant`, and
`optical_size` targets. Unavailable continuous targets are `None`.

`location` is an ordinary `dict[str, float]` mapping OpenType axis tags to the
values used to load the glyph.

Indices are Python integers and continuous targets are floats. An unavailable
continuous target is `None`.

`GlyphData` objects compare by identity. Compare tensor payloads explicitly when
content equality is required.

Define a local `DataLoader.collate_fn` for the model's input contract and use
[`pad_outlines`](#pad-outlines) when its payloads are variable-length outlines.
See [DataLoader](../guide/basic/dataloader.md).

### `GlyphIdData`

`GlyphIdData` is what `LoadGlyph` returns for a `GlyphIdSample`. It carries the
same payload, glyph reference, resolved variation location, and `font_idx`,
`weight`, `width`, `italic`, `slant`, and `optical_size` targets as
`GlyphData`, without the `codepoint` and `character_idx` targets that a glyph
no character maps to cannot have.

It follows the same rules as `GlyphData`: identity equality, integer indices,
and `None` for an unavailable continuous target.

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

### `TYPE_DIM: int`

Number of element types. Current value: `7`.

### `COORD_DIM: int`

Coordinates width. Current value: `6` (`[cx0, cy0, cx1, cy1, x, y]`).
