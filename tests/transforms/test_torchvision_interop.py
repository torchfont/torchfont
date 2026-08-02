import pytest
import torch

from torchfont.structures import FontRef, GlyphRef
from torchfont.tf_tensors import Bitmap
from torchfont.transforms import (
    Compose,
    HorizontalFlip,
    LoadGlyph,
    RenderBitmap,
    ToPureTensor,
)

tv_tensors = pytest.importorskip("torchvision.tv_tensors")
v2 = pytest.importorskip("torchvision.transforms.v2")

FONT = "tests/fonts/roboto/Roboto[wdth,wght].ttf"


def _render() -> Bitmap:
    pipeline = Compose([LoadGlyph(), HorizontalFlip(), RenderBitmap(size=64)])
    return pipeline(GlyphRef(FontRef(FONT, 0), ord("A"), {"wght": 700.0}))


def test_pure_tensor_feeds_torchvision_v2() -> None:
    pure = ToPureTensor()(_render())
    assert type(pure) is torch.Tensor

    pipeline = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((32, 32), antialias=True),
            v2.RandomHorizontalFlip(p=1.0),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    out = pipeline(pure.unsqueeze(0))

    assert isinstance(out, tv_tensors.Image)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32


def test_bitmap_wraps_as_tv_image() -> None:
    image = tv_tensors.Image(_render())
    resized = v2.Resize((16, 16), antialias=True)(image)

    assert isinstance(resized, tv_tensors.Image)
    assert resized.shape == (1, 16, 16)


def test_torchvision_passes_label_through() -> None:
    image, label = v2.RandomHorizontalFlip(p=1.0)(
        tv_tensors.Image(_render()), torch.tensor(7)
    )

    assert isinstance(image, tv_tensors.Image)
    assert int(label) == 7
