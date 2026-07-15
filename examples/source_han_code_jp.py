from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import load_glyph


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor]:
    return load_glyph(sample.ref)


def collate_fn(
    batch: list[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor]:
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


def main() -> None:
    dataset = GlyphDataset(
        root="data/adobe/source-han-code-jp",
        patterns=("OTC/*.ttc",),
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

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
