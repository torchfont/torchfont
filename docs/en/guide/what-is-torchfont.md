# What is TorchFont

TorchFont is a **PyTorch library for treating font glyph outlines as
machine-learning tensors**. Instead of rasterizing glyphs into images first, it
works directly with path
commands such as move, line, quadratic, and cubic segments.

::: info
TorchFont is an unofficial library and is not affiliated with the PyTorch
project.
:::

## In One Minute

- **Unified dataset API**:
  `FontFolder` (local), `FontRepo` (arbitrary Git), and `GoogleFonts`.
- **Training-ready output**:
  `dataset[i] -> (types, coords, style_idx, content_idx)`.
- **Fast preprocessing**:
  Rust backend (`skrifa` + PyO3) reduces Python-side overhead.
- **DataLoader-friendly**:
  worker processes rebuild native backend state after pickle/unpickle.

## Problems It Solves

Common pain points in font ML workflows:

- data collection pipelines differ per project and are hard to reproduce
- rasterization-heavy preprocessing makes experiments harder to compare
- static and variable fonts are often handled with separate logic

TorchFont standardizes collection, tensorization, and labeling so you can spend
more time on model design.

## How It Works

- **Dataset layer**
  - `FontFolder`: scans local directories for fonts
  - `FontRepo`: synchronizes a Git repository, then indexes fonts like
    `FontFolder`
  - `GoogleFonts`: preset configuration of `FontRepo` for Google Fonts
- **Rust backend**
  - maps charmap codepoints to glyphs
  - converts outlines into command sequences + 6D coordinate sequences
  - normalizes coordinates by `units_per_em`
  - keeps quadratic and cubic Beziers as distinct command types
- **Transform layer**
  - `QuadToCubic`: normalize `QUAD_TO` into `CURVE_TO`
  - `LimitSequenceLength`: truncate long sequences
  - `Patchify`: reshape into fixed-length patches
  - `Compose`: chain transforms in order

## Minimal Example

```python
from torchfont.datasets import FontFolder

# root must exist
# e.g. root="~/fonts" (or "tests/fonts" if you cloned this repository)
dataset = FontFolder(root="~/fonts")

types, coords, style_idx, content_idx = dataset[0]
print(types.shape)         # (seq_len,)
print(coords.shape)        # (seq_len, 6)
print(style_idx, content_idx)
```

## When It Is a Good Fit

- you want to disentangle content (character) and style (font instance)
- you want to run experiments on large collections including variable fonts
- you want a consistent vector-first representation for
  generation/classification/representation learning

## Read Next

- [Quickstart](/en/guide/getting-started)
- [Glyph Data Format](/en/guide/glyph-data-format)
- [DataLoader Integration](/en/guide/dataloader)
