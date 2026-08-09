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
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

TorchFont is an **unofficial** library based on PyTorch for deep learning with vector fonts.
It is not affiliated with or endorsed by the PyTorch project.

TorchFont is local-first: point `GlyphDataset` at a font directory or a
repository checkout that already exists on disk, and TorchFont turns font files
into lightweight glyph references. Load outlines explicitly with `LoadGlyph`
in your transform pipeline when tensors are needed.

## Installation

The package requires Python 3.10+ and PyTorch 2.4+.

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
import math

import torch
from torch.utils.data import DataLoader

from torchfont import GlyphData, Outline, pad_outlines
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph


def collate_fn(samples: list[GlyphData[Outline]]):
    return {
        "outline": pad_outlines([sample.data for sample in samples]),
        "font_idx": torch.tensor(
            [sample.font_idx for sample in samples], dtype=torch.long
        ),
        "weight": torch.tensor(
            [
                math.nan if sample.weight is None else sample.weight
                for sample in samples
            ],
            dtype=torch.float32,
        ),
    }


dataset = GlyphDataset(
    root="~/fonts",  # or "tests/fonts" in this repository
    patterns=("*.ttf",),
    codepoints=range(0x20, 0x7F),  # printable ASCII
    transform=LoadGlyph(),
)

loader = DataLoader(
    dataset,
    batch_size=8,
    shuffle=True,
    collate_fn=collate_fn,
)
batch = next(iter(loader))

print(batch["outline"].shape)  # (8, L)
print(batch["outline"].coords.shape)  # (8, L, 6)
print(batch["weight"].shape)  # (8,)
```

Define `collate_fn` locally to choose the targets and missing-value representation
required by your model. Use `pad_outlines` to pad variable-length outlines.

## What TorchFont Focuses On

- local font directories and repository checkouts as the input boundary
- local font indexing plus explicit outline loading through lightweight glyph references
- `torchvision.transforms.v2`-style semantic pipelines for adapting glyph samples
- PyTorch `DataLoader` integration through an explicit, customizable `collate_fn`

Manage font repository synchronization with Git or another tool, then point
`GlyphDataset(root=...)` at the resulting directory.

## Citation

If you find TorchFont useful in your work, please consider citing the following BibTeX entry:

```bibtex
@software{fujioka2025torchfont,
    author = {Fujioka, Takumu},
    title  = {{TorchFont: A Machine Learning library for Vector Fonts}},
    year   = {2025},
    url    = {https://github.com/torchfont/torchfont}
}
```
