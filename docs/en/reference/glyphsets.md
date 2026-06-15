# Glyphsets API

<!-- markdownlint-disable MD013 -->

Definitions of codepoint groups from the Google Fonts "glyphsets" repository.
These groups can be used to filter the glyphs in a font to only those that are
relevant for a particular script, or to limit the number of glyphs in a font to
reduce its size.

```python
from torchfont.glyphsets import LATIN_CORE, LATIN_KERNEL, get_glyphset_codepoints
```

## `LATIN_CORE`

```python
from torchfont.glyphsets import LATIN_CORE
```

Codepoints of the **GF Latin Core** glyphset as a list of integers. This
glyphset covers the basic Latin characters needed for most Western European
languages.

## `LATIN_KERNEL`

```python
from torchfont.glyphsets import LATIN_KERNEL
```

Codepoints of the **GF Latin Kernel** glyphset as a list of integers. This
glyphset covers a smaller subset of Latin characters suitable for minimal font
configurations.

## `get_glyphset_codepoints`

```python
from torchfont.glyphsets import get_glyphset_codepoints
```

```python
codepoints = get_glyphset_codepoints(glyphset_name)
```

Returns the codepoints of a named glyphset from the Google Fonts glyphset
registry as a list of integers.

| Parameter      | Type   | Description                          |
| -------------- | ------ | ------------------------------------ |
| `glyphset_name` | `str` | name of the glyphset to look up      |

| Return value | Type           | Description                           |
| ------------ | -------------- | ------------------------------------- |
| `codepoints` | `list[int]`    | codepoints in the named glyphset      |

Raises `ValueError` if the glyphset name is not found in the registry.

### Example

```python
codepoints = get_glyphset_codepoints("GF_Latin_Core")
```
