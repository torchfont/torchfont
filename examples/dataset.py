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

    print(f"{len(dataset)=}")
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.character_classes)=}")

    for i in tqdm(range(len(dataset)), desc="Iterating over datasets"):
        _ = dataset[i]


if __name__ == "__main__":
    main()
