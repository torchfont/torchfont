import torch
from torch import Tensor
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import patchify, render_bitmap
from torchfont.utils import collate_outline


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor, Tensor]:
    types = sample.types[:512]
    coords = sample.coords[:512]
    bitmap = render_bitmap(types, coords)
    patch_types, patch_coords = patchify(types, coords, patch_size=32)
    return patch_types, patch_coords, bitmap


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    outline_pairs = [(pt, pc) for pt, pc, _ in batch]
    bitmaps = torch.stack([bitmap for _, _, bitmap in batch])
    types_t, coords_t = collate_outline(outline_pairs)
    return types_t, coords_t, bitmaps


def main() -> None:
    dataset = GlyphDataset(
        root="data/google/fonts",
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
        transform=transform,
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

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
