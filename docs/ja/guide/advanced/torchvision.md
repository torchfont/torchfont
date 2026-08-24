# TorchVision との連携

TorchFont はビットマップを通常のテンソルとして返すため、TorchVision の Transform へ直接
渡せます。

## TorchVision をインストールする

TorchVision が PyTorch と同じ uv インデックスを使用するように設定します。環境に合う
バリアントを選び、`pyproject.toml` に追加します。

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

その後、TorchVision を追加します。

```bash
uv add torchvision
```

## 画像パイプラインを構築する

`RenderBitmap` より前にアウトラインの Transform を適用し、その後に TorchVision の
Transform でチャンネル変換、リサイズ、dtype 変換を行います。

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

`RenderBitmap` は `H x W` の `uint8` テンソルを生成します。`T.ToImage` はチャンネル次元を
追加し、`T.ToDtype(..., scale=True)` は画素値を `[0, 255]` から `[0, 1]` へ変換します。
`T.ToPureTensor` はモデルへ渡す前に TorchVision の画像ラッパーを取り除きます。

パイプラインは `GlyphData` のメタデータを維持し、`data` の内容だけを置き換えます。
ローカルな `collate_fn` で画像テンソルをスタックし、モデルが必要とするターゲットを
選択できます。

```python
def collate_fn(samples):
    return {
        "image": torch.stack([sample.data for sample in samples]),
        "character_idx": torch.tensor(
            [sample.character_idx for sample in samples], dtype=torch.long
        ),
    }
```
