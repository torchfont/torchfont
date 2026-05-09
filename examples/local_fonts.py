import dataclasses

from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import quad_to_cubic
from torchfont.utils import collate_fn


def normalize_curves(sample: GlyphSample) -> GlyphSample:
    types, coords = quad_to_cubic(sample.types, sample.coords)
    return dataclasses.replace(sample, types=types, coords=coords)


def main() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoints=range(0x20, 0x7F),
        transform=normalize_curves,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))

    print(f"{len(dataset)=}")
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{batch.types.shape=}")
    print(f"{batch.coords.shape=}")


if __name__ == "__main__":
    main()
