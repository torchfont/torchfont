# Subset Extraction

Use `dataset.targets` to build style/content-based subsets efficiently.

## Prerequisite

The examples below assume a constructed TorchFont dataset. For a quick start,
you can use:

```python
import torch
from torch.utils.data import Subset
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="tests/fonts",
    patterns=("*.ttf",),
    codepoints=range(0x80),
)

t = dataset.targets
if t.numel() == 0:
    raise ValueError("dataset is empty; relax patterns/codepoints first")
print(t.shape)  # (N, 2)
```

If you are using a checked-out font repository instead, only `root` and
`patterns` need to change.

- `t[:, 0]`: `style_idx`
- `t[:, 1]`: `content_idx`

## Pick labels that are guaranteed to exist

Using labels from an observed sample keeps the combined condition example
realistic.

```python
style_idx = t[0, 0].item()
content_idx = t[0, 1].item()
```

## Common filters

```python
# by style
style_indices = torch.where(t[:, 0] == style_idx)[0].tolist()
style_subset = Subset(dataset, style_indices)

# by content
content_indices = torch.where(t[:, 1] == content_idx)[0].tolist()
content_subset = Subset(dataset, content_indices)

# by style + content
mask = (t[:, 0] == style_idx) & (t[:, 1] == content_idx)
combined_indices = torch.where(mask)[0].tolist()
subset = Subset(dataset, combined_indices)
```

## Filter by names

Use `style_classes` / `content_classes` for human-readable filtering:

```python
style_name = dataset.style_classes[style_idx]
style_indices = [
    idx for idx, name in enumerate(dataset.style_classes) if name == style_name
]
style_tensor = torch.as_tensor(style_indices, dtype=torch.long)
style_mask = torch.isin(t[:, 0], style_tensor)

char = dataset.content_classes[content_idx]
content_idx_from_name = dataset.content_class_to_idx[char]

mask = style_mask & (t[:, 1] == content_idx_from_name)
indices = torch.where(mask)[0].tolist()
subset = Subset(dataset, indices)
```

## Complete script

The runnable script name is `examples/subset_by_targets.py`.
For the full example list, see [Example Gallery](/en/examples/).
