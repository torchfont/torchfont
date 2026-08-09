# Quickstart

## Install uv

PyTorch is distributed across separate indexes depending on target hardware: CPU, CUDA, ROCm, and more. Normally you have to specify the right index manually, making reproducibility hard to guarantee. With uv, you configure the index once in `pyproject.toml` and everyone installs the correct build with a single command.

Follow the [uv](https://docs.astral.sh/uv/) [installation guide](https://docs.astral.sh/uv/getting-started/installation/) to install it.

## Using a PyTorch index

Add the PyTorch index and source to `pyproject.toml`. Choose the variant that matches your environment:

::: code-group

```toml [CPU-only]
[[tool.uv.index]]
name = "pytorch-cpu"
url = "https://download.pytorch.org/whl/cpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cpu" }]
```

```toml [CUDA 11.8]
[[tool.uv.index]]
name = "pytorch-cu118"
url = "https://download.pytorch.org/whl/cu118"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu118" }]
```

```toml [CUDA 12.6]
[[tool.uv.index]]
name = "pytorch-cu126"
url = "https://download.pytorch.org/whl/cu126"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu126" }]
```

```toml [CUDA 12.8]
[[tool.uv.index]]
name = "pytorch-cu128"
url = "https://download.pytorch.org/whl/cu128"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu128" }]
```

```toml [CUDA 13.0]
[[tool.uv.index]]
name = "pytorch-cu130"
url = "https://download.pytorch.org/whl/cu130"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-cu130" }]
```

```toml [ROCm 7.2]
[[tool.uv.index]]
name = "pytorch-rocm72"
url = "https://download.pytorch.org/whl/rocm7.2"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-rocm72" }]
```

```toml [Intel GPUs]
[[tool.uv.index]]
name = "pytorch-xpu"
url = "https://download.pytorch.org/whl/xpu"
explicit = true

[tool.uv.sources]
torch = [{ index = "pytorch-xpu" }]
```

:::

For multi-platform setups, see the [uv PyTorch integration guide](https://docs.astral.sh/uv/guides/integration/pytorch/).

## Installing dependencies

Add PyTorch and TorchFont together:

```bash
uv add torch torchfont
```

## Verify the installation

This creates a dataset by scanning the current directory for fonts and prints the number of samples. With no fonts present it prints `0`. If it runs without error, the installation is complete.

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root=".")
print(len(dataset))
```
