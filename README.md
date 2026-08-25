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
[![PyTorch](https://img.shields.io/badge/PyTorch-2.5+-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)

TorchFont is an **unofficial** library based on PyTorch for deep learning with vector fonts.
It is not affiliated with or endorsed by the PyTorch project.

TorchFont is local-first: point `CodepointDataset` or `GlyphIdDataset` at a font
directory or a repository checkout that already exists on disk, and TorchFont
turns font files into lightweight glyph references. `CodepointDataset` indexes one
sample per face and codepoint, while `GlyphIdDataset` indexes one sample per
face and glyph, reaching ligatures and alternates no codepoint maps to. Load
outlines explicitly with `LoadGlyph` in your transform pipeline when tensors are
needed.

## Installation

The package requires Python 3.10+ and PyTorch 2.5+.

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
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont import CodepointData, ElementType, Outline
from torchfont.datasets import CodepointDataset
from torchfont.transforms import LoadGlyph


def collate_fn(samples: list[CodepointData[Outline]]):
    outlines = [sample.data for sample in samples]
    return {
        "types": pad_sequence(
            [outline.types for outline in outlines],
            batch_first=True,
            padding_value=ElementType.PAD,
        ),
        "coords": pad_sequence(
            [outline.coords for outline in outlines], batch_first=True
        ),
    }


dataset = CodepointDataset(
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

print(batch["types"].shape)  # (8, L)
print(batch["coords"].shape)  # (8, L, 6)
```

## What TorchFont Focuses On

- local font directories and repository checkouts as the input boundary
- local font indexing by codepoint or by glyph id, plus explicit outline loading through lightweight glyph references
- `torchvision.transforms.v2`-style semantic pipelines for adapting glyph samples
- PyTorch `DataLoader` integration through an explicit, customizable `collate_fn`

Manage font repository synchronization with Git or another tool, then point
`CodepointDataset(root=...)` or `GlyphIdDataset(root=...)` at the resulting
directory.

## Citing TorchFont

If you find TorchFont useful in your work, please consider citing the following BibTeX entry:

```bibtex
@software{fujioka2025torchfont,
    author = {Fujioka, Takumu},
    title  = {{TorchFont}: A Machine Learning Library for Vector Fonts},
    year   = {2025},
    url    = {https://github.com/torchfont/torchfont}
}
```
