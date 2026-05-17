# Glyph Data Format

TorchFont datasets return each glyph sample as a `GlyphSample`:

```python
sample = dataset[i]
```

| Field                | Type                | Shape          | Meaning                          |
| -------------------- | ------------------- | -------------- | -------------------------------- |
| `sample.types`       | `torch.LongTensor`  | `(seq_len,)`   | Element type sequence            |
| `sample.coords`      | `torch.FloatTensor` | `(seq_len, 6)` | Coordinates per path element     |
| `sample.style_idx`   | `int`               | scalar         | Style class ID                   |
| `sample.content_idx` | `int`               | scalar         | Content class ID                 |
| `sample.metrics`     | `torch.FloatTensor` | `(15,)`        | Per-glyph and font-level metrics |
| `sample.glyph_name`  | `str`               | —              | PostScript glyph name            |

`GlyphSample` is a dataclass; field access by name is the intended API.

## Outline model

- **Outline**: a sequence of path elements
- **Subpath**: a sequence of path elements starting with `MoveTo` and ending with `Close`; open subpaths (without a closing `Close`) are also treated as subpaths for convenience
- **Path element**: one element type plus one coordinates row
- **Element type**: `MoveTo`, `LineTo`, `QuadTo`, `CurveTo`, `Close`, `End`, or `Pad`
- **Coordinates**: `[cx0, cy0, cx1, cy1, x, y]`

`sample.metrics` is a 1-D `float32` tensor of shape `(15,)`:

```python
# [adv_w, lsb, x_min, y_min, x_max, y_max,
#  ascent, descent, leading, cap_height, x_height, avg_width,
#  italic_angle, units_per_em, is_monospace]
m = sample.metrics
```

Values are UPM-normalised where applicable; `nan` when missing.

## `types`

```python
from torchfont.io import ElementType

print(ElementType.QUAD_TO, ElementType.QUAD_TO.value)
# ElementType.QUAD_TO 3
```

- `ElementType.END` marks end of sequence
- `ElementType.PAD` is mainly introduced by `pad_sequence` or custom padding

## `coords`

Each step uses a 6D vector:

```text
[cx0, cy0, cx1, cy1, x, y]
```

- `ElementType.MOVE_TO` / `ElementType.LINE_TO`: control points are zero;
  endpoint `(x, y)` is used
- `ElementType.QUAD_TO`: one control point `(cx0, cy0)` and endpoint `(x, y)`
  are used (`cx1, cy1` are zero)
- `ElementType.CURVE_TO`: two control points `(cx0, cy0)` and `(cx1, cy1)`,
  and the endpoint
  `(x, y)` are used
- `ElementType.CLOSE` / `ElementType.END` / `ElementType.PAD`: zeros

::: info
Coordinates are normalized by the font `units_per_em`.
:::

## Quadratic Bezier handling

Quadratic curves are emitted as `ElementType.QUAD_TO` without conversion. To
keep tensor shape fixed, `ElementType.QUAD_TO` uses
`[cx0, cy0, 0, 0, x, y]`.

## Style and content labels

### `style_idx`

- static fonts: usually `Family + Subfamily` (e.g. `Lato Regular`)
- variable fonts: one class per named instance when available
- variable fonts with empty instance names: `Family` only
- variable fonts without named instances: fallback to `Family + Subfamily` (or
  `Family`)

```python
metadata = dataset.metadata

print(dataset.style_classes[:5])
print(metadata.styles[:5])
print(metadata.style_name_to_idxs)
```

`metadata.styles` exposes source-based collision-safe IDs derived from relative
font path / face / instance information, while
`metadata.style_name_to_idxs` keeps every index for duplicate display names.

### `content_idx`

- one class per Unicode character

```python
metadata = dataset.metadata

print(dataset.content_classes[:5])
print(metadata.contents[:5])
print(metadata.content_id_to_idx)
```

## `targets`

```python
t = dataset.targets  # shape: (N, 2), dtype: torch.long

style_all = t[:, 0]    # column 0: style_idx
content_all = t[:, 1]  # column 1: content_idx
```

## Shapes after utilities

`quad_to_cubic` preserves both `types` and `coords` shapes.

`patchify` changes the shape: a sequence of length `N` becomes
`(num_patches, patch_size)` for `types` and `(num_patches, patch_size, 6)` for
`coords`. If you add custom dataset transforms for model-specific shaping, keep
`style_idx` and `content_idx` aligned with the returned sample.
