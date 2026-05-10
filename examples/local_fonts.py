from torch import Tensor
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.utils import collate_outline


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor]:
    return sample.types[:512], sample.coords[:512]


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
        collate_fn=collate_outline,
    )

    types_t, coords_t = next(iter(dataloader))

    print(f"{len(dataset)=}")
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{types_t.shape=}")
    print(f"{coords_t.shape=}")


if __name__ == "__main__":
    main()
