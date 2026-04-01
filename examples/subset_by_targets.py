import torch
from torch.utils.data import Subset

from torchfont.datasets import GlyphDataset


def main() -> None:
    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoint_filter=range(0x80),
    )

    print(f"{len(dataset)=}")
    print(f"{dataset.targets.shape=}")

    t = dataset.targets

    # Subset by style
    style_idx = t[0, 0].item()
    style_indices = torch.where(t[:, 0] == style_idx)[0].tolist()
    style_sub = Subset(dataset, style_indices)
    print(f"style={style_idx}: {len(style_sub)} samples")

    # Subset by content
    content_idx = t[0, 1].item()
    content_indices = torch.where(t[:, 1] == content_idx)[0].tolist()
    content_sub = Subset(dataset, content_indices)
    print(f"content={content_idx}: {len(content_sub)} samples")

    # Subset by style & content
    mask = (t[:, 0] == style_idx) & (t[:, 1] == content_idx)
    combined_indices = torch.where(mask)[0].tolist()
    combined_sub = Subset(dataset, combined_indices)
    print(f"style={style_idx} & content={content_idx}: {len(combined_sub)} samples")


if __name__ == "__main__":
    main()
