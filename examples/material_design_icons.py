from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn


def main() -> None:
    dataset = GlyphDataset(
        root="data/google/material_design_icons",
        patterns=("variablefont/*.ttf",),
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
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")

    for batch in dataloader:
        print(batch)
        break


if __name__ == "__main__":
    main()
