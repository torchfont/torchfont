from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont import CodepointData, Outline, pad_outlines
from torchfont import transforms as T
from torchfont.datasets import CodepointDataset


def collate_fn(batch: list[CodepointData[Outline]]) -> Outline:
    return pad_outlines([data.data for data in batch])


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
