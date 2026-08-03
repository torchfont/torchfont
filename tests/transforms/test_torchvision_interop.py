import pytest
import torch

from torchfont import FontRef, GlyphData, GlyphRef, GlyphSample
from torchfont.transforms import (
    Compose,
    HorizontalFlip,
    LoadGlyph,
    RenderBitmap,
)

tv_tensors = pytest.importorskip("torchvision.tv_tensors")
v2 = pytest.importorskip("torchvision.transforms.v2")

FONT = "tests/fonts/roboto/Roboto[wdth,wght].ttf"


def _render() -> torch.Tensor:
    pipeline = Compose([LoadGlyph(), HorizontalFlip(), RenderBitmap(size=64)])
    return pipeline(GlyphRef(FontRef(FONT, 0), ord("A")))


def test_to_image_converts_bitmap_to_channel_first_tv_image() -> None:
    pipeline = v2.Compose(
        [
            v2.ToImage(),
            v2.Resize((32, 32), antialias=True),
            v2.RandomHorizontalFlip(p=1.0),
            v2.ToDtype(torch.float32, scale=True),
        ]
    )
    out = pipeline(_render())

    assert isinstance(out, tv_tensors.Image)
    assert out.shape == (1, 32, 32)
    assert out.dtype == torch.float32


def test_render_bitmap_returns_a_plain_tensor() -> None:
    assert type(_render()) is torch.Tensor


def test_torchvision_pipeline_preserves_glyph_data_to_model_boundary() -> None:
    ref = GlyphRef(FontRef(FONT, 0), ord("A"))
    sample = GlyphSample(ref, font_idx=3, character_idx=5)
    pipeline = Compose(
        [
            LoadGlyph(),
            RenderBitmap(size=64),
            v2.ToImage(),
            v2.Resize((32, 32), antialias=True),
            v2.ToDtype(torch.float32, scale=True),
            v2.ToPureTensor(),
        ]
    )

    out = pipeline(sample)

    assert isinstance(out, GlyphData)
    assert out.ref is sample.ref
    assert out.font_idx == sample.font_idx
    assert out.character_idx == sample.character_idx
    assert type(out.data) is torch.Tensor
    assert out.data.shape == (1, 32, 32)
    assert out.data.dtype == torch.float32
    assert out.data.min() >= 0.0
    assert out.data.max() <= 1.0


def test_torchvision_passes_label_through() -> None:
    image, label = v2.RandomHorizontalFlip(p=1.0)(
        v2.ToImage()(_render()), torch.tensor(7)
    )

    assert isinstance(image, tv_tensors.Image)
    assert int(label) == 7
