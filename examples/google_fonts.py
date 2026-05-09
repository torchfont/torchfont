import dataclasses

from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import limit_sequence_length, patchify
from torchfont.utils import collate_fn


def transform(sample: GlyphSample) -> GlyphSample:
    types, coords = limit_sequence_length(sample.types, sample.coords, max_len=512)
    types, coords = patchify(types, coords, patch_size=32)
    return dataclasses.replace(sample, types=types, coords=coords)


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

    for _ in tqdm(dataloader, desc="Iterating over datasets"):
        pass


if __name__ == "__main__":
    main()
