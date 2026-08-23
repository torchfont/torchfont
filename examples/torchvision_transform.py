import math
from typing import TYPE_CHECKING, cast

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torchvision.transforms import v2 as T
from tqdm import tqdm

from torchfont import GlyphSample
from torchfont import transforms as FT
from torchfont.datasets import GlyphDataset
from torchfont.glyphsets import LATIN_CORE

if TYPE_CHECKING:
    from torchfont import GlyphData, Outline


class TransformPipeline(torch.nn.Module):
    """Build two model inputs from one shared outline pipeline."""

    def __init__(self) -> None:
        super().__init__()
        self.prepare_outline = FT.Compose(
            [
                FT.LoadGlyph(location="random"),
                FT.RemoveOverlaps(),
                FT.QuadToCubic(merge_curves=True),
                FT.RandomAffine(degrees=5.0, translate=(0.05, 0.05)),
            ]
        )
        self.rasterize = FT.Compose(
            [
                FT.RenderBitmap(size=96),
                T.ToImage(),
                T.Resize((64, 64), antialias=True),
                T.ToDtype(torch.float32, scale=True),
                T.ToPureTensor(),
            ]
        )

    def forward(
        self, sample: GlyphSample
    ) -> tuple[Tensor, Tensor, Tensor, int, int, float | None]:
        data = cast("GlyphData[Outline]", self.prepare_outline(sample))
        image_data = cast("GlyphData[Tensor]", self.rasterize(data))
        return (
            data.data.types,
            data.data.coords,
            image_data.data,
            data.font_idx,
            data.character_idx,
            data.weight,
        )


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor, int, int, float | None]],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
    types, coords, images, fonts, characters, weights = zip(*batch, strict=True)
    return (
        pad_sequence(list(types), batch_first=True),
        pad_sequence(list(coords), batch_first=True),
        torch.stack(images),
        torch.tensor(fonts, dtype=torch.long),
        torch.tensor(characters, dtype=torch.long),
        torch.tensor(
            [math.nan if weight is None else weight for weight in weights],
            dtype=torch.float32,
        ),
    )


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
        transform=TransformPipeline(),
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
