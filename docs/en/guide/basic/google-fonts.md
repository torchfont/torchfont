# Google Fonts Setup

## Why Google Fonts

[Google Fonts](https://fonts.google.com/) provides a large collection of font
families covering many styles and writing systems.

- **Broad coverage**: The repository contains many families, styles, and writing
  systems.
- **Repository layout**: Font files are grouped under the `apache`, `ofl`, and
  `ufl` directories used throughout this guide.
- **Versioned data**: Git commits can identify the exact revision used for an
  experiment.

Review the license included with each font before using or redistributing it.

## Clone the repository

Clone Google Fonts into the path used by the examples in this guide. `--depth 1`
fetches only the latest commit to keep the download size small.

```bash
git clone --depth 1 https://github.com/google/fonts.git data/google/fonts
```
