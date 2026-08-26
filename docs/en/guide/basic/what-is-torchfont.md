# What is TorchFont

::: info
TorchFont is an unofficial library and is not affiliated with the PyTorch
project.
:::

## Why machine learning for fonts?

Developing a font demands substantial time and effort, especially for writing
systems with large character sets. Machine learning can support several parts
of that work:

- **Font generation**: synthesizing new typefaces or interpolating smoothly between existing ones
- **Style transfer**: applying the aesthetic of one typeface to the glyphs of another
- **Classification and retrieval**: identifying fonts from images or finding visually similar typefaces
- **Digitization**: reconstructing outlines from scanned specimens of historical or rare type

## Features

- **Outline-first representation**:
  TorchFont exposes scalable vector outlines directly instead of requiring an
  intermediate bitmap representation.
- **Fast on-the-fly processing**:
  the Rust backend reads font files during dataset access, so pipelines can work
  from the original files without a required preprocessing format.
- **Composable transforms**:
  an `Outline`-aware class API provides `torchvision.transforms.v2`-style `Transform`,
  `Compose`, `RandomApply`, `RandomChoice`, and `RandomOrder` building blocks for
  reusable data pipelines.
  Deterministic functionals remain available as low-level kernels.
