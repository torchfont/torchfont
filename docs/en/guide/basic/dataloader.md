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
the exact input contract of your model. Use PyTorch's `pad_sequence` for the
outline tensors and tensorize only the targets the model needs:

```python
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import CodepointDataset
from torchfont import CodepointData, ElementType, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(samples: list[CodepointData[Outline]]):
    outlines = [sample.data for sample in samples]
    return {
        "types": pad_sequence(
            [outline.types for outline in outlines],
            batch_first=True,
            padding_value=ElementType.PAD,
        ),
        "coords": pad_sequence(
            [outline.coords for outline in outlines], batch_first=True
        ),
        "lengths": torch.tensor([len(outline) for outline in outlines]),
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

print(batch["types"].shape)
print(batch["coords"].shape)
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

Padded elements use `ElementType.PAD`. Build the boolean mask expected by
attention modules directly from the padded element types:

```python
mask = batch["types"] == ElementType.PAD  # (64, 369), True where padding
```

Keep the original lengths when the padded tensors must be restored with
PyTorch's `unpad_sequence`:

```python
from torch.nn.utils.rnn import unpad_sequence

types = unpad_sequence(batch["types"], batch["lengths"], batch_first=True)
coords = unpad_sequence(batch["coords"], batch["lengths"], batch_first=True)
```

`torchfont.nn` modules take the padded tensors directly:

```python
from torchfont.nn import OutlineEmbedding

tokens = OutlineEmbedding(embedding_dim=256)(batch["types"], batch["coords"])

print(tokens.shape)  # (64, 369, 256)
```
