# Glyph Data Format

TorchFont datasets return each glyph sample as:

```python
types, coords, style_idx, content_idx = dataset[i]
```

| Element       | Type                | Shape          | Meaning              |
| ------------- | ------------------- | -------------- | -------------------- |
| `types`       | `torch.LongTensor`  | `(seq_len,)`   | Pen command sequence |
| `coords`      | `torch.FloatTensor` | `(seq_len, 6)` | Coordinate sequence  |
| `style_idx`   | `int`               | scalar         | Style class ID       |
| `content_idx` | `int`               | scalar         | Content class ID     |

## `types`

```python
from torchfont.io.outline import CommandType

print(CommandType.QUAD_TO, CommandType.QUAD_TO.value)
# CommandType.QUAD_TO 3
```

- `CommandType.END` marks end of sequence
- `CommandType.PAD` is mainly introduced by `pad_sequence` or `Patchify`

## `coords`

Each step uses a 6D vector:

```text
[cx0, cy0, cx1, cy1, x, y]
```

- `CommandType.MOVE_TO` / `CommandType.LINE_TO`: control points are zero;
  endpoint `(x, y)` is used
- `CommandType.QUAD_TO`: one control point `(cx0, cy0)` and endpoint `(x, y)`
  are used (`cx1, cy1` are zero)
- `CommandType.CURVE_TO`: two control points `(cx0, cy0)` and `(cx1, cy1)`,
  and the endpoint
  `(x, y)` are used
- `CommandType.CLOSE` / `CommandType.END` / `CommandType.PAD`: zeros

::: info
Coordinates are normalized by the font `units_per_em`.
:::

## Quadratic Bezier handling

Quadratic curves are emitted as `CommandType.QUAD_TO` without conversion. To
keep tensor shape fixed, `CommandType.QUAD_TO` uses
`[cx0, cy0, 0, 0, x, y]`.

## Style and content labels

### `style_idx`

- static fonts: usually `Family + Subfamily` (e.g. `Lato Regular`)
- variable fonts: one class per named instance when available
- variable fonts with empty instance names: `Family` only
- variable fonts without named instances: fallback to `Family + Subfamily` (or
  `Family`)

```python
print(dataset.style_classes[:5])
print(dataset.style_class_to_idx)
```

::: warning
`style_classes` can include duplicate names. In that case,
`style_class_to_idx` keeps only the last occurrence of each name.
When duplicate-safe filtering is required, enumerate `style_classes` directly.
:::

### `content_idx`

- one class per Unicode character

```python
print(dataset.content_classes[:5])
print(dataset.content_class_to_idx)
```

## `targets`

```python
t = dataset.targets  # shape: (N, 2), dtype: torch.long

style_all = t[:, 0]    # column 0: style_idx
content_all = t[:, 1]  # column 1: content_idx
```

## Shapes after transforms

With `Patchify`:

- before: `types=(seq_len,)`, `coords=(seq_len, 6)`
- after: `types=(num_patches, patch_size)`,
  `coords=(num_patches, patch_size, 6)`

`style_idx` and `content_idx` stay unchanged.
