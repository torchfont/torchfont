import dataclasses

from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.utils import collate_fn


def transform(sample: GlyphSample) -> GlyphSample:
    return dataclasses.replace(
        sample, types=sample.types[:512], coords=sample.coords[:512]
    )


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
