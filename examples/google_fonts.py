import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.glyphsets import LATIN_CORE
from torchfont.instance_fn import grid_instances
from torchfont.transforms import (
    load_glyph,
    patchify,
    quad_to_cubic,
    remove_overlaps,
    render_bitmap,
)


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor, Tensor]:
    types, coords = load_glyph(sample.ref)
    types = types[:512]
    coords = coords[:512]
    types, coords = remove_overlaps(types, coords)
    types, coords = quad_to_cubic(types, coords, merge_curves=True)
    bitmap = render_bitmap(types, coords)
    patch_types, patch_coords = patchify(types, coords, patch_size=32)
    return patch_types, patch_coords, bitmap


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
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{len(dataset.character_classes)=}")

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
