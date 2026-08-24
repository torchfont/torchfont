# Dataset Integration

Font repositories can be added to a project as Git submodules. Git records the
selected revision while keeping the font files outside the project's own
history.

## Add a repository

Choose a path under `data/` and add the repository:

```bash
git submodule add --depth 1 https://github.com/google/fonts.git data/google/fonts
```

Commit the generated `.gitmodules` file and the recorded submodule revision.
Initialize the checkout after cloning the parent repository:

```bash
git submodule update --init --depth 1
```

Point `CodepointDataset` at the submodule and select its font files with `patterns`:

```python
from torchfont.datasets import CodepointDataset

dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=("apache/*/*.ttf", "ofl/*/*.ttf", "ufl/*/*.ttf"),
)
```

## Font repositories

### Google Fonts

The [Google Fonts repository](https://github.com/google/fonts) contains a large
collection of font families. Its top-level `apache`, `ofl`, and `ufl`
directories group binary font files by license.

```python
dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=("apache/*/*.ttf", "ofl/*/*.ttf", "ufl/*/*.ttf"),
)
```

### Material Design Icons

The [Material Design Icons repository](https://github.com/google/material-design-icons)
contains the classic Material Icons font files in several styles. It also
contains Material Symbols variable fonts.

```bash
git submodule add --depth 1 \
    https://github.com/google/material-design-icons.git \
    data/google/material-design-icons
```

```python
dataset = CodepointDataset(
    root="data/google/material-design-icons",
    patterns=("font/*.ttf", "font/*.otf", "variablefont/*.ttf"),
)
```

### Font Awesome

The [Font Awesome repository](https://github.com/FortAwesome/Font-Awesome)
contains its free regular, solid, and brand icon fonts under `otfs/`.

```bash
git submodule add --depth 1 \
    https://github.com/FortAwesome/Font-Awesome.git \
    data/fortawesome/font-awesome
```

```python
dataset = CodepointDataset(
    root="data/fortawesome/font-awesome",
    patterns=("otfs/*.otf",),
)
```

### Source Han Code JP

[Source Han Code JP](https://github.com/adobe-fonts/source-han-code-jp) provides
Japanese fonts as individual OpenType fonts and as an OpenType Collection.
`CodepointDataset` expands every face in the collection; see
[Font Collections](./font-collections.md) for details.

```bash
git submodule add --depth 1 \
    https://github.com/adobe-fonts/source-han-code-jp.git \
    data/adobe/source-han-code-jp
```

```python
dataset = CodepointDataset(
    root="data/adobe/source-han-code-jp",
    patterns=("OTC/*.ttc",),
)
```

Review the license files in each repository before using or redistributing its
fonts.
