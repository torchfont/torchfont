import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import Compose, LimitSequenceLength, Patchify, render_bitmap
from torchfont.utils import GlyphBatch
from torchfont.utils import collate_fn as base_collate_fn

_pipeline = Compose([LimitSequenceLength(max_len=512), Patchify(patch_size=32)])


def collate_fn(
    batch: list[GlyphSample],
) -> tuple[GlyphBatch, Tensor]:
    bitmaps = torch.stack([render_bitmap(s.types, s.coords, size=64) for s in batch])
    return base_collate_fn([_pipeline(s) for s in batch]), bitmaps


def main() -> None:
    dataset = GlyphDataset(
        root="data/google/fonts",
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
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
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")

    for _ in tqdm(dataloader, desc="Iterating over datasets"):
        pass


if __name__ == "__main__":
    main()
