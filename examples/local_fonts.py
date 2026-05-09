import dataclasses

from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.utils import collate_fn


def transform(sample: GlyphSample) -> GlyphSample:
    return dataclasses.replace(
        sample, types=sample.types[:512], coords=sample.coords[:512]
    )


def main() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoints=range(0x20, 0x7F),
        transform=transform,
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
