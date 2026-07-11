# Batching with DataLoader

## Why use a DataLoader

Neural network training processes data in batches rather than one sample at a
time. Batching stabilizes gradient estimates and makes full use of GPU
parallelism. `DataLoader` is PyTorch's standard utility that handles batch
construction, shuffling, and parallel loading.

## Define a `transform`

`GlyphSample` carries a glyph reference and target indices. Which tensors you
load depends on the task, so use `transform` to call `load_glyph` and keep only
the values you need.

Like PyTorch datasets, `GlyphDataset` has a `transform` argument that applies a
transformation to each item. Define a function that loads `types` and `coords`
from `sample.ref`, pass it to the dataset, and verify the output. Run the
following code:

```python
from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import load_glyph


def transform(sample: GlyphSample):
    return load_glyph(sample.ref)


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=transform,
)

types, coords = dataset[0]

print(types.shape)
print(coords.shape)
```

With `transform`, `dataset[0]` now returns a `(types, coords)` tuple instead of
a `GlyphSample`. `1` is the sequence length of this glyph; it varies per glyph.
You will see output like:

```
torch.Size([1])
torch.Size([1, 6])
```

## Create a DataLoader

Glyph outline sequences are variable-length, so batching requires a `collate_fn`.
Use `pad_sequence` to align sequences within a batch. Run the following code:

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import load_glyph


def transform(sample: GlyphSample):
    return load_glyph(sample.ref)


def collate_fn(batch):
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=transform,
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
processes. Long sequences increase transfer overhead, so the `transform` truncates
each sequence to the first 512 elements. Use `tqdm` to iterate over all batches
and measure throughput. Run the following code:

```python
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import load_glyph


def transform(sample: GlyphSample):
    types, coords = load_glyph(sample.ref)
    return types[:512], coords[:512]


def collate_fn(batch):
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=transform,
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

You will see output like the following. `it/s` is the batch processing speed.
The entire Google Fonts dataset of 12.4 million samples completes in just 2
minutes, fast enough for practical training loops.

```
len(dataset)=12460609
100%|██████████| 194698/194698 [02:03<00:00, 1570.64it/s]
```
