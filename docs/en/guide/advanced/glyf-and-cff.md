# glyf and CFF Outlines

OpenType fonts commonly encode outlines in one of two forms:

| Outline table | Curve representation | Variable-font form |
| --- | --- | --- |
| TrueType `glyf` | Quadratic Bézier curves | `glyf` with `gvar` |
| PostScript CFF | Cubic Bézier curves | CFF2 |

The `.ttf` and `.otf` extensions are useful conventions but are not the API
boundary in TorchFont. The outline table determines whether a loaded curve is
represented by `ElementType.QUAD_TO` or `ElementType.CURVE_TO`.

## One transform interface

`LoadGlyph` converts both formats to the same `Outline` type. Lines, subpaths,
coordinates, metadata, batching, geometric transforms, and rendering therefore
use the same APIs:

```python
from torchfont import transforms as T

transform = T.Compose(
    [
        T.LoadGlyph(location="random"),
        T.RemoveOverlaps(),
        T.RandomAffine(degrees=5.0),
        T.RenderBitmap(size=64),
    ]
)
```

The pipeline accepts static or variable fonts backed by `glyf`, CFF, or CFF2.
No format branch is needed when the selected transforms support both quadratic
and cubic segments.

## Normalizing the curve representation

Some models require every outline to use the same curve degree. Convert at the
start of the outline pipeline when that is part of the model's input contract:

```python
# Use cubic curves for every font.
cubic_transform = T.Compose([T.LoadGlyph(), T.QuadToCubic(merge_curves=True)])

# Or use quadratic curves for every font.
quadratic_transform = T.Compose([T.LoadGlyph(), T.CubicToQuad()])
```

`QuadToCubic` converts each quadratic segment exactly. `CubicToQuad` may replace
one cubic segment with several quadratic segments to approximate it within
about `1e-3` em. Keep the original representation when the model supports both
element types; normalize only when a uniform representation is useful.
