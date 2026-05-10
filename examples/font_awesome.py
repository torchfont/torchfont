from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.utils import collate_outline


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor]:
    return sample.types, sample.coords


def main() -> None:
    dataset = GlyphDataset(
        root="data/fortawesome/font-awesome",
        patterns=("otfs/*.otf",),
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=collate_outline,
    )

    print(f"{len(dataset)=}")
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
