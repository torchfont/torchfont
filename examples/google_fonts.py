from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset
from torchfont.glyphsets import LATIN_CORE
from torchfont.instance_fn import grid_instances
from torchfont.transforms import (
    Compose,
    LoadGlyph,
    QuadToCubic,
    RemoveOverlaps,
    RenderBitmap,
)

if TYPE_CHECKING:
    from torchfont.structures import GlyphData, GlyphSample, Outline


class PrepareGlyph:
    """Build two model inputs from one shared outline pipeline."""

    def __init__(self) -> None:
        self.outline = Compose(
            [LoadGlyph(), RemoveOverlaps(), QuadToCubic(merge_curves=True)]
        )
        self.render = RenderBitmap()

    def __call__(self, sample: GlyphSample) -> tuple[Tensor, Tensor, Tensor]:
        data = cast("GlyphData[Outline]", self.outline(sample))
        bitmap = self.render(data).data
        return data.data.types, data.data.coords, bitmap


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    types = pad_sequence([types for types, _, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords, _ in batch], batch_first=True)
    bitmaps = torch.stack([bitmap for _, _, bitmap in batch])
    return types, coords, bitmaps


def main() -> None:
    dataset = GlyphDataset(
        codepoints=LATIN_CORE,
        root="data/google/fonts",
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
        instance_fn=grid_instances({"wght": 7, "wdth": 3, "opsz": 3, "slnt": 2}),
        transform=PrepareGlyph(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=collate_fn,
    )

    print(f"{len(dataset)=}")
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{len(dataset.character_classes)=}")

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
