from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.transforms import GlyphData, LoadGlyph, Outline

transform = LoadGlyph()


def collate_fn(
    batch: list[GlyphData[Outline]],
) -> tuple[Tensor, Tensor]:
    types = pad_sequence([data.data.types[:512] for data in batch], batch_first=True)
    coords = pad_sequence([data.data.coords[:512] for data in batch], batch_first=True)
    return types, coords


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

    types_t, coords_t = next(iter(dataloader))

    print(f"{len(dataset)=}")
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{len(dataset.character_classes)=}")
    print(f"{types_t.shape=}")
    print(f"{coords_t.shape=}")


if __name__ == "__main__":
    main()
