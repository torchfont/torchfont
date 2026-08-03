# Batching with DataLoader

## Why use a DataLoader

Neural network training processes data in batches rather than one sample at a
time. Batching stabilizes gradient estimates and makes full use of GPU
parallelism. `DataLoader` is PyTorch's standard utility that handles batch
construction, shuffling, and parallel loading.

## Define a `transform`

`GlyphSample` carries a glyph reference and target indices. Use `LoadGlyph` as
the first pipeline transform to load its semantic `Outline` while retaining the
sample metadata.

Like PyTorch datasets, `GlyphDataset` has a `transform` argument that applies a
transformation to each item. Pass `LoadGlyph()` directly and verify the output:

```python
from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import LoadGlyph


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

data: GlyphData[Outline] = dataset[0]
types, coords = data.data.types, data.data.coords

print(types.shape)
print(coords.shape)
```

With `LoadGlyph`, `dataset[0]` returns `GlyphData[Outline]`. Its `data` field is
the outline and its `sample` field retains the original metadata. The first
shape is `(N,)` and the second is `(N, 6)`, where `N` varies by glyph. For
example:

```
torch.Size([37])
torch.Size([37, 6])
```

## Create a DataLoader

Glyph outline sequences are variable-length, so batching requires a `collate_fn`.
Use `pad_sequence` to align sequences within a batch. Run the following code:

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(batch: list[GlyphData[Outline]]):
    types = pad_sequence([item.data.types for item in batch], batch_first=True)
    coords = pad_sequence([item.data.coords for item in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
types_t, coords_t = next(iter(loader))

print(types_t.shape)
print(coords_t.shape)
```

`collate_fn` pads each sequence to the length of the longest one in the batch.
The first dimension is the batch size. The second dimension is the longest
sequence length in the batch and varies per batch. You will see output like:

```
torch.Size([64, 369])
torch.Size([64, 369, 6])
```

## Multi-process loading

Set `num_workers` and `prefetch_factor` to load data in parallel worker
processes. Long sequences increase transfer overhead, so this example's
`collate_fn` truncates each sequence to the first 512 elements. Use `tqdm` to
iterate over all batches and measure throughput. Run the following code:

```python
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(batch: list[GlyphData[Outline]]):
    types = pad_sequence([item.data.types[:512] for item in batch], batch_first=True)
    coords = pad_sequence([item.data.coords[:512] for item in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(
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
    num_workers=8,
    prefetch_factor=2,
)

print(f"{len(dataset)=}")

for batch in tqdm(loader):
    pass
```

The dataset length depends on the selected font files and their character maps.
The progress bar reports batch throughput as `it/s`; use it to choose worker and
prefetch settings for your storage and training environment.

```
len(dataset)=...
100%|██████████| .../... [..., ...it/s]
```
