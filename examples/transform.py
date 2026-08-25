from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont import CodepointData, ElementType, Outline
from torchfont import transforms as T
from torchfont.datasets import CodepointDataset


def collate_fn(batch: list[CodepointData[Outline]]) -> tuple[Tensor, Tensor]:
    outlines = [data.data for data in batch]
    return (
        pad_sequence(
            [outline.types for outline in outlines],
            batch_first=True,
            padding_value=ElementType.PAD,
        ),
        pad_sequence([outline.coords for outline in outlines], batch_first=True),
    )


def main() -> None:
    transform = T.Compose(
        [
            T.LoadGlyph(location="random"),
            T.RemoveOverlaps(),
            T.QuadToCubic(merge_curves=True),
        ]
    )

    dataset = CodepointDataset(
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
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.character_classes)=}")

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
