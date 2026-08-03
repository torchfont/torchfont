from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import RandomLocation


def collate_fn(
    batch: list[GlyphData[Outline]],
) -> tuple[Tensor, Tensor, list[dict[str, float]]]:
    types = pad_sequence([data.data.types[:512] for data in batch], batch_first=True)
    coords = pad_sequence([data.data.coords[:512] for data in batch], batch_first=True)
    locations = [dict(data.location) for data in batch]
    return types, coords, locations


def main() -> None:
    dataset = GlyphDataset(
        root="data/google/fonts",
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
        transform=RandomLocation(),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    types_t, coords_t, locations = next(iter(dataloader))

    print(f"{len(dataset)=}")
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.character_classes)=}")
    print(f"{types_t.shape=}")
    print(f"{coords_t.shape=}")
    print(f"{locations=}")


if __name__ == "__main__":
    main()
