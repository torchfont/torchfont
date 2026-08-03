from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.transforms import v2
from tqdm import tqdm

from torchfont.datasets import GlyphDataset
from torchfont.glyphsets import LATIN_CORE
from torchfont.transforms import (
    Compose,
    LoadGlyph,
    QuadToCubic,
    RandomAffine,
    RemoveOverlaps,
    RenderBitmap,
)

if TYPE_CHECKING:
    from torchfont.structures import GlyphData, GlyphSample, Outline


class GlyphPipeline(torch.nn.Module):
    """Build two model inputs from one shared outline pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.prepare_outline = Compose(
            [
                LoadGlyph(),
                RemoveOverlaps(),
                QuadToCubic(merge_curves=True),
                RandomAffine(degrees=5.0, translate=(0.05, 0.05)),
            ]
        )
        self.rasterize = Compose(
            [
                RenderBitmap(size=96),
                v2.ToImage(),
                v2.Resize((64, 64), antialias=True),
                v2.ToDtype(torch.float32, scale=True),
                v2.ToPureTensor(),
            ]
        )

    def forward(self, sample: GlyphSample) -> tuple[Tensor, Tensor, Tensor]:
        data = cast("GlyphData[Outline]", self.prepare_outline(sample))
        image_data = cast("GlyphData[Tensor]", self.rasterize(data))
        return data.data.types, data.data.coords, image_data.data


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    types = pad_sequence([types for types, _, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords, _ in batch], batch_first=True)
    images = torch.stack([image for _, _, image in batch])
    return types, coords, images


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
        transform=GlyphPipeline(),
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
    print(f"{len(dataset.character_classes)=}")

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
