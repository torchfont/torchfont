import dataclasses

from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import quad_to_cubic
from torchfont.utils import collate_fn


def normalize_curves(sample: GlyphSample) -> GlyphSample:
    types, coords = quad_to_cubic(sample.types, sample.coords)
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
        transform=normalize_curves,
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
