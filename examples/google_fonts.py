from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GoogleFonts
from torchfont.transforms import (
    Compose,
    LimitSequenceLength,
    Patchify,
)

transform = Compose(
    (
        LimitSequenceLength(max_len=512),
        Patchify(patch_size=32),
    ),
)

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    transform=transform,
    download=True,
)


def collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    types_list = [types for types, _, _, _ in batch]
    coords_list = [coords for _, coords, _, _ in batch]
    style_label_list = [style for _, _, style, _ in batch]
    content_label_list = [content for _, _, _, content in batch]

    types_tensor = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords_tensor = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    style_label_tensor = torch.as_tensor(style_label_list, dtype=torch.long)
    content_label_tensor = torch.as_tensor(content_label_list, dtype=torch.long)

    return types_tensor, coords_tensor, style_label_tensor, content_label_tensor


dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=8,
    prefetch_factor=2,
    collate_fn=collate_fn,
)

print(f"{len(dataset)=}")
print(f"{len(dataset.content_classes)=}")
print(f"{len(dataset.style_classes)=}")

for batch in tqdm(dataloader, desc="Iterating over datasets"):
    sample = batch
