import torch
from torch.utils.data import Subset

from torchfont.datasets import GlyphDataset


def main() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoints=range(0x80),
    )

    print(f"{len(dataset)=}")
    print(f"{len(dataset.font_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{len(dataset.character_classes)=}")
    print(f"{dataset.style_targets.shape=}")
    print(f"{dataset.character_targets.shape=}")

    style_targets = dataset.style_targets
    character_targets = dataset.character_targets

    # Subset by style
    style_idx = style_targets[0].item()
    style_indices = torch.where(style_targets == style_idx)[0].tolist()
    style_sub = Subset(dataset, style_indices)
    print(f"style={style_idx}: {len(style_sub)} samples")

    # Subset by character
    character_idx = character_targets[0].item()
    character_indices = torch.where(character_targets == character_idx)[0].tolist()
    character_sub = Subset(dataset, character_indices)
    print(f"character={character_idx}: {len(character_sub)} samples")

    # Subset by style & character
    mask = (style_targets == style_idx) & (character_targets == character_idx)
    combined_indices = torch.where(mask)[0].tolist()
    combined_sub = Subset(dataset, combined_indices)
    print(f"style={style_idx} & character={character_idx}: {len(combined_sub)} samples")


if __name__ == "__main__":
    main()
