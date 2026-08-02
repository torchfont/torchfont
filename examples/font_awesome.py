from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(
    batch: list[GlyphData[Outline]],
) -> tuple[Tensor, Tensor]:
    types = pad_sequence([data.data.types for data in batch], batch_first=True)
    coords = pad_sequence([data.data.coords for data in batch], batch_first=True)
    return types, coords


def main() -> None:
    dataset = GlyphDataset(
        root="data/fortawesome/font-awesome",
        patterns=("otfs/*.otf",),
        transform=LoadGlyph(),
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

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
