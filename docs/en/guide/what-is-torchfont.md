# What is TorchFont

::: info
TorchFont is an unofficial library and is not affiliated with the PyTorch
project.
:::

## Why Font Machine Learning?

Developing a font demands substantial time and effort. A Latin family with a few weights may take months; a CJK font covering tens of thousands of characters can take years. That cost is precisely why most of the world's writing systems remain poorly served by existing type libraries. Machine learning offers a path to scaling that effort:

- **Font generation**: synthesizing new typefaces or interpolating smoothly between existing ones
- **Style transfer**: applying the aesthetic of one typeface to the glyphs of another
- **Classification and retrieval**: identifying fonts from images or finding visually similar typefaces
- **Digitization**: reconstructing outlines from scanned specimens of historical or rare type

Fonts for minority languages in particular offer very limited choices, and that situation has changed little over the years. Reducing development costs is one of the most direct ways to change it.

## Features

- **Outline-first representation**:
  fonts are used at many scales, making bitmap-based generation of limited practical value.
  TorchFont provides font outlines, the native data format, as a dataset.
- **Fast on-the-fly processing**:
  the Rust backend reads font files directly at training time, fast enough to require no preprocessing step.
  Font files remain the single source of truth.
- **Freely composable transforms**:
  rather than a class-based compose pattern like torchvision's `transforms.Compose`,
  TorchFont provides utility functions that you wire together yourself,
  giving you full control over how font data is prepared for your model.
