# Batching with DataLoader

## Why use a DataLoader

Neural network training commonly processes several samples as a batch.
`DataLoader` is PyTorch's standard utility for batch construction, shuffling,
and parallel loading.

## Define a `transform`

`CodepointSample` carries a glyph reference and target indices. Use `LoadGlyph` as
the first pipeline transform to load its semantic `Outline` while retaining the
sample metadata.

Like PyTorch datasets, `CodepointDataset` has a `transform` argument that applies a
transformation to each item. Pass `LoadGlyph()` directly and verify the output:

```python
from torchfont.datasets import CodepointDataset
from torchfont import CodepointData, Outline
from torchfont.transforms import LoadGlyph


dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

data: CodepointData[Outline] = dataset[0]

print(data.data.shape)
print(data.data.coords.shape)
```

With `LoadGlyph`, `dataset[0]` returns `CodepointData[Outline]`. Its `data` field
the outline, while its other fields retain the reference, location, and targets.
The first shape is `(N,)` and the second is `(N, 6)`, where `N` varies by glyph.
For example:

```
torch.Size([37])
torch.Size([37, 6])
```

## Create a DataLoader

Glyph outline sequences are variable-length, so define a local `collate_fn` for
the exact input contract of your model. Use `pad_outlines` for the payload and
tensorize only the targets the model needs:

```python
import math

import torch
from torch.utils.data import DataLoader

from torchfont.datasets import CodepointDataset
from torchfont import CodepointData, Outline, pad_outlines
from torchfont.transforms import LoadGlyph


def collate_fn(samples: list[CodepointData[Outline]]):
    return {
        "outline": pad_outlines([sample.data for sample in samples]),
        "font_idx": torch.tensor(
            [sample.font_idx for sample in samples], dtype=torch.long
        ),
        "weight": torch.tensor(
            [
                math.nan if sample.weight is None else sample.weight
                for sample in samples
            ],
            dtype=torch.float32,
        ),
    }


dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
)
batch = next(iter(loader))

print(batch["outline"].shape)
print(batch["outline"].coords.shape)
print(batch["weight"].shape)
```

Outlines are padded to the length of the longest one in the batch. The first
dimension is the batch size; the second is the longest sequence length in the
batch and varies per batch. Targets become 1-dimensional tensors of length
`batch_size`. You will see output like:

```
torch.Size([64, 369])
torch.Size([64, 369, 6])
torch.Size([64])
```

## Work with a padded batch

Padded elements use `ElementType.PAD`. Rather than recovering them by comparing
against that value, read `padding_mask`, which is exactly what attention modules
expect as `key_padding_mask`:

```python
mask = batch["outline"].padding_mask  # (64, 369), True where padding
```

`unpad_outlines()` explicitly splits a padded batch back into the single
outlines that went in. By contrast, `Outline.unbind()` preserves padding just
like an ordinary tensor operation:

```python
from torchfont import unpad_outlines

singles = unpad_outlines(batch["outline"])

print(len(singles), singles[0].shape)
```

`torchfont.nn` modules take an `Outline`, batched or not, so a padded batch goes
straight into a model:

```python
from torchfont.nn import OutlineEmbedding

tokens = OutlineEmbedding(embedding_dim=256)(batch["outline"])

print(tokens.shape)  # (64, 369, 256)
```

## Batch outlines without a DataLoader

`pad_outlines` applies the same padding step directly:

```python
from torchfont import pad_outlines

batched = pad_outlines([dataset[0].data, dataset[1].data])

print(batched.shape)
```
