# What is TorchFont

TorchFont is a **PyTorch library for treating font glyph outlines as
machine-learning tensors**. It keeps path commands such as move, line,
quadratic, and cubic segments available directly while also emitting a small
bitmap view for default samples.

::: info
TorchFont is an unofficial library and is not affiliated with the PyTorch
project.
:::

## In One Minute

- **Primary dataset API**:
  `GlyphDataset(root=...)` reads any local font directory or repository
  checkout already on disk.
- **Sample-first output**:
  `dataset[i] -> GlyphSample(types, coords, bitmap, style_idx, content_idx, metrics, glyph_name)`.
- **Built-in batching**:
  `torchfont.utils.collate_fn(batch) -> GlyphBatch`.
- **Fast preprocessing**:
  Rust backend (`skrifa` + PyO3) reduces Python-side overhead.
- **DataLoader-friendly**:
  worker processes rebuild native backend state after pickle/unpickle.

## Problems It Solves

Common pain points in font ML workflows:

- the boundary between "how fonts are collected" and "how glyphs are read" is
  often blurry
- image preprocessing pipelines often diverge across experiments
- static and variable fonts are often handled with separate logic

TorchFont standardizes tensorization, labeling, and batching so you can spend
more time on model design.

## How It Works

- **Dataset layer**
  - `GlyphDataset`: scans local directories for fonts
  - a Git repository is just another input directory once it is checked out
- **Rust backend**
  - maps charmap codepoints to glyphs
  - converts outlines into command sequences + 6D coordinate sequences
  - renders a fixed 64 x 64 grayscale bitmap for each default sample
  - normalizes coordinates by `units_per_em`
  - keeps quadratic and cubic Beziers as distinct command types
- **Transform layer**
  - `QuadToCubic`: normalize `QUAD_TO` into `CURVE_TO`
  - `LimitSequenceLength`: truncate long sequences
  - `Patchify`: reshape into fixed-length patches
  - `Compose`: chain transforms in order
- **Batching utilities**
  - `collate_fn`: pads variable-length samples into `GlyphBatch`
  - `GlyphBatch.targets`: style and content indices as `(B, 2)` tensor
  - `GlyphBatch.metrics`: per-sample metrics as `(B, 15)` float tensor

## Minimal Example

```python
from torchfont.datasets import GlyphDataset

# root must exist
# e.g. root="~/fonts" (or "tests/fonts" if you cloned this repository)
dataset = GlyphDataset(root="~/fonts")

sample = dataset[0]
print(sample.types.shape)         # (seq_len,)
print(sample.coords.shape)        # (seq_len, 6)
print(sample.bitmap.shape)        # (64, 64)
print(sample.style_idx, sample.content_idx)
print(sample.glyph_name)
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
