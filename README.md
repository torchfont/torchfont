# TorchFont

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="docs/public/brand/torchfont-logo-dark.svg">
    <source media="(prefers-color-scheme: light)" srcset="docs/public/brand/torchfont-logo-light.svg">
    <img alt="TorchFont logo" src="docs/public/brand/torchfont-logo-light.svg" width="640">
  </picture>
</p>

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/pypi/pyversions/torchfont)](https://pypi.org/project/torchfont/)
[![CI](https://github.com/torchfont/torchfont/actions/workflows/ci.yml/badge.svg)](https://github.com/torchfont/torchfont/actions)
[![Documentation](https://readthedocs.org/projects/torchfont/badge/?version=latest)](https://torchfont.readthedocs.io/)
[![PyPI Downloads](https://static.pepy.tech/personalized-badge/torchfont?period=total&units=INTERNATIONAL_SYSTEM&left_color=GRAY&right_color=GREEN&left_text=downloads)](https://pepy.tech/projects/torchfont)
[![PyPI version](https://img.shields.io/pypi/v/torchfont)](https://pypi.org/project/torchfont/)
[![Rust](https://img.shields.io/badge/Rust-2024-orange?logo=rust)](https://www.rust-lang.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

TorchFont is an **unofficial** library based on PyTorch for deep learning with vector fonts.
It is not affiliated with or endorsed by the PyTorch project.

TorchFont is local-first: point `GlyphDataset` at a font directory or a
repository checkout that already exists on disk, and TorchFont turns glyph
outlines into `GlyphSample` / `GlyphBatch` objects for PyTorch pipelines.

## Installation

The package requires Python 3.10+ and PyTorch 2.3+.

Install TorchFont with **uv**:

```bash
uv add torchfont
```

Or with **pip**:

```bash
pip install torchfont
```

## Quickstart

```python
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn

dataset = GlyphDataset(
    root="~/fonts",  # or "tests/fonts" in this repository
    patterns=("*.ttf",),
    codepoints=range(0x20, 0x7F),  # printable ASCII
)

loader = DataLoader(dataset, batch_size=8, shuffle=True, collate_fn=collate_fn)
batch = next(iter(loader))

print(batch.types.shape)
print(batch.coords.shape)
print(batch.targets.shape)
print(batch.metrics.shape)
```

## What TorchFont Focuses On

- local font directories and repository checkouts as the input boundary
- Rust-backed outline decoding into `GlyphSample` with outline tensors, metrics, and glyph name
- sample-first transforms such as `QuadToCubic`, `LimitSequenceLength`, and `Patchify`
- DataLoader integration through `GlyphBatch` and `collate_fn`

TorchFont does not need to own Git clone / fetch / checkout in the main
workflow. Sync repositories with Git or another tool, then point
`GlyphDataset(root=...)` at the resulting directory.

## Citation

If you find TorchFont useful in your work, please consider citing the following BibTeX entry:

```bibtex
@software{fujioka2025torchfont,
    title        = {TorchFont: A Machine Learning library for Vector Fonts},
    author       = {Takumu Fujioka},
    year         = 2025,
    journal      = {GitHub repository},
    publisher    = {GitHub},
    howpublished = {\url{https://github.com/torchfont/torchfont}}
}
```
