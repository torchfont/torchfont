# TorchVision Integration

TorchFont returns bitmaps as ordinary tensors, so they can be passed directly
to TorchVision transforms.

## Install TorchVision

Configure TorchVision to use the same uv index as PyTorch. Choose the variant
that matches your environment and add it to `pyproject.toml`:

::: code-group

```toml [CPU-only]
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
torchvision = [{ index = "pytorch-cpu" }]
```

```toml [CUDA 11.8]
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu118" }]
torchvision = [{ index = "pytorch-cu118" }]
```

```toml [CUDA 12.6]
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
torchvision = [{ index = "pytorch-cu126" }]
```

```toml [CUDA 12.8]
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
torchvision = [{ index = "pytorch-cu128" }]
```

```toml [CUDA 13.0]
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
torchvision = [{ index = "pytorch-cu130" }]
```

```toml [ROCm 7.2]
[[tool.uv.index]]
name = "pytorch-rocm72"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-rocm72" }]
torchvision = [{ index = "pytorch-rocm72" }]
```

```toml [Intel GPUs]
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-xpu" }]
torchvision = [{ index = "pytorch-xpu" }]
```

:::

Then add TorchVision:

```bash
uv add torchvision
```

## Build an image pipeline

Apply outline transforms before `RenderBitmap`, then use TorchVision transforms
for channel conversion, resizing, and dtype conversion:

```python
import torch
from torchvision.transforms import v2 as T

from torchfont import transforms as FT
from torchfont.datasets import CodepointDataset

transform = FT.Compose(
    [
        FT.LoadGlyph(location="random"),
        FT.RemoveOverlaps(),
        FT.RandomAffine(degrees=5.0, translate=(0.05, 0.05)),
        FT.RenderBitmap(size=96),
        T.ToImage(),
        T.Resize((64, 64), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.ToPureTensor(),
    ]
)

dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=("apache/*/*.ttf", "ofl/*/*.ttf", "ufl/*/*.ttf"),
    transform=transform,
)

sample = dataset[0]
print(sample.data.shape)  # (1, 64, 64)
print(sample.data.dtype)  # torch.float32
```

`RenderBitmap` produces an `H x W` `uint8` tensor. `T.ToImage` adds the channel
dimension, and `T.ToDtype(..., scale=True)` converts pixel values from
`[0, 255]` to `[0, 1]`. `T.ToPureTensor` removes the TorchVision image wrapper at
the model boundary.

The pipeline preserves the `GlyphData` metadata and replaces only its `data`
payload. A local `collate_fn` can stack the image payloads and select the targets
required by the model:

```python
def collate_fn(samples):
    return {
        "image": torch.stack([sample.data for sample in samples]),
        "character_idx": torch.tensor(
            [sample.character_idx for sample in samples], dtype=torch.long
        ),
    }
```
