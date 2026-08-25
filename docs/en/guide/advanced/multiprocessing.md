# Multiprocess Data Loading

Set `num_workers` and `prefetch_factor` to load data in parallel worker
processes.

Each batch is padded to its longest outline, so a single very large glyph
inflates the whole batch and the transfer to the training process with it. This
example caps every outline at 512 elements in its local `collate_fn`. Define it
at module level: worker processes pickle the `collate_fn`, so a lambda will not
work.

Use `tqdm` to iterate over all batches and measure throughput:

```python
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchfont import CodepointData, ElementType, Outline
from torchfont.datasets import CodepointDataset
from torchfont.transforms import LoadGlyph

MAX_ELEMENTS = 512


def collate_fn(samples: list[CodepointData[Outline]]):
    outlines = [sample.data[:MAX_ELEMENTS] for sample in samples]
    return {
        "types": pad_sequence(
            [outline.types for outline in outlines],
            batch_first=True,
            padding_value=ElementType.PAD,
        ),
        "coords": pad_sequence(
            [outline.coords for outline in outlines], batch_first=True
        ),
        "font_idx": torch.tensor(
            [sample.font_idx for sample in samples], dtype=torch.long
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
    num_workers=8,
    prefetch_factor=2,
)

print(f"{len(dataset)=}")

for batch in tqdm(loader):
    pass
```

The progress bar reports batch throughput as `it/s`; use it to choose worker and
prefetch settings for your storage and training environment.

::: tip Padding without truncation
Truncation drops geometry. To keep whole outlines and still avoid the padding
cost, group glyphs of similar length with a length-aware `Sampler` instead of
capping their length.
:::
