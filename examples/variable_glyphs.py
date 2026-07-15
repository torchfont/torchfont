from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import VariableGlyphDataset, VariableGlyphSample
from torchfont.instance_fn import grid_instance_count
from torchfont.transforms import load_glyph, random_location


def transform(
    sample: VariableGlyphSample,
) -> tuple[Tensor, Tensor, dict[str, float]]:
    location = random_location(sample.ref.font)
    types, coords = load_glyph(sample.ref, location)
    return types[:512], coords[:512], location


def collate_fn(
    batch: list[tuple[Tensor, Tensor, dict[str, float]]],
) -> tuple[Tensor, Tensor, list[dict[str, float]]]:
    types = pad_sequence([types for types, _, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords, _ in batch], batch_first=True)
    locations = [location for _, _, location in batch]
    return types, coords, locations


def main() -> None:
    dataset = VariableGlyphDataset(
        root="data/google/fonts",
        patterns=(
            "apache/*/*.ttf",
            "ofl/*/*.ttf",
            "ufl/*/*.ttf",
            "!ofl/adobeblank/*.ttf",
        ),
        instance_fn=grid_instance_count({"wght": 7, "wdth": 3, "opsz": 3, "slnt": 2}),
        transform=transform,
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
