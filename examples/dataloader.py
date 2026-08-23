from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset


def main() -> None:
    dataset = GlyphDataset(
        root="data/google/fonts",
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
    )

    dataloader = DataLoader(
        dataset,
        batch_size=64,
        shuffle=True,
        num_workers=8,
        prefetch_factor=2,
        collate_fn=list,
    )

    print(f"{len(dataset)=}")
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.character_classes)=}")

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
