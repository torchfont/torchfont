from torch.utils.data import DataLoader
from tqdm import tqdm

from torchfont.datasets import GlyphDataset
from torchfont.transforms import Compose, LimitSequenceLength, Patchify
from torchfont.utils import collate_fn


def main() -> None:
    transform = Compose(
        (
            LimitSequenceLength(max_len=512),
            Patchify(patch_size=32),
        ),
    )

    dataset = GlyphDataset(
        root="data/google/fonts",
        patterns=("ofl/*/*.ttf",),
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
    print(f"{len(dataset.content_classes)=}")
    print(f"{len(dataset.style_classes)=}")

    for batch in tqdm(dataloader, desc="Iterating over datasets"):
        _ = batch


if __name__ == "__main__":
    main()
