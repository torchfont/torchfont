from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.transforms import Compose, LimitSequenceLength, Patchify
from torchfont.utils import collate_fn


def main() -> None:
    transform = Compose(
        (
            LimitSequenceLength(max_len=256),
            Patchify(patch_size=32),
        ),
    )

    dataset = GlyphDataset(
        root="tests/fonts",
        patterns=("*.ttf",),
        codepoints=range(0x20, 0x7F),
        transform=transform,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=8,
        shuffle=True,
        collate_fn=collate_fn,
    )

    batch = next(iter(dataloader))

    print(f"{len(dataset)=}")
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")
    print(f"{batch.types.shape=}")
    print(f"{batch.coords.shape=}")
    print(f"{batch.mask.shape=}")


if __name__ == "__main__":
    main()
