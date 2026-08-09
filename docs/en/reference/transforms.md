# Transforms

TorchFont provides composable transforms for loading, modifying, and rendering
glyphs. TorchVision is optional and can be added when the result should enter an
image pipeline.

## Data types

```python
from torchfont import GlyphData, Outline
```

`Outline(types, coords)` keeps the two coupled tensors together.
`GlyphData[T]` keeps a transformed payload, glyph reference, variation location,
and targets together. A pipeline can change its payload from `Outline` to a
bitmap tensor without losing the other fields.

## Loading and composition

```python
from torchfont.transforms import (
    Compose,
    LoadGlyph,
    RandomApply,
    RandomSplitSegments,
    RemoveOverlaps,
)

transform = Compose(
    [
        LoadGlyph(),
        RandomApply(RandomSplitSegments(split_probability=0.2), p=0.5),
        RemoveOverlaps(),
    ]
)

data = transform(sample)
outline = data.data
```

`LoadGlyph` loads one `GlyphSample` or `GlyphRef`. A sample becomes
`GlyphData[Outline]`, while a bare reference becomes `Outline`.
`LoadGlyph` uses the face's default location unless `location="random"` is set.
The random policy samples one location for a `GlyphSample` or `GlyphRef` and
records it in `GlyphData.location`; on a static face it naturally uses an empty
location. For dataset samples, it also resolves the parallel `weight`, `width`,
`italic`, `slant`, and `optical_size` targets on the returned `GlyphData`.

Transforms accept nested inputs and preserve their structure. Corresponding
outlines in one call receive the same randomly sampled parameters. Apply the
transform separately to independent samples. Random transforms use PyTorch's default RNG, so
`torch.manual_seed` and DataLoader worker seeding apply normally.
Built-in transforms can be used with multiprocessing data loaders. `Compose`
accepts `nn.Module` transforms; define custom transforms as `nn.Module`
subclasses rather than plain callables. An empty `Compose` leaves its input
unchanged. Use `Compose` inside `RandomApply` to group several transforms.

Calling `eval()` does not disable random data augmentation. Use a deterministic
pipeline for evaluation.

`RandomApply(transform, p)` controls whether one transform is applied.
Probabilities such as `RandomSplitSegments.split_probability` control behavior
inside an already-applied transform.

## Built-in transforms

| Category | Transforms |
| --- | --- |
| Loading | `LoadGlyph` |
| Containers | `Compose`, `RandomApply` |
| Curves | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| Outline | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| Subpaths | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| Geometry | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| Output | `RenderBitmap` |

`RenderBitmap` changes each `Outline` leaf into a plain `uint8` tensor. When
these leaves are inside `GlyphData`, its reference, location, and targets remain
alongside the converted payload.

### Using rendered glyphs with TorchVision

`RenderBitmap` returns a plain grayscale `H x W` tensor. Use
`torchvision.transforms.v2.ToImage()` as the boundary into an image pipeline:
it adds the channel
dimension and returns a `tv_tensors.Image` of shape `1 x H x W`. Fields in an
enclosing `GlyphData` are preserved.
`RenderBitmap(antialias=False)` produces binary edge coverage. This controls
vector rasterization and is independent of the `antialias` option on a later
`v2.Resize`, which controls image resampling.

```python
import torch
from torchvision.transforms import v2

from torchfont.transforms import Compose, LoadGlyph, RenderBitmap

transform = Compose(
    [
        LoadGlyph(),
        RenderBitmap(size=96),
        v2.ToImage(),
        v2.RandomAffine(degrees=(-5.0, 5.0), translate=(0.05, 0.05), fill=0),
        v2.Resize((64, 64), antialias=True),
        v2.ToDtype(torch.float32, scale=True),
        v2.ToPureTensor(),
    ]
)

data = transform(sample)
image = data.data  # Tensor, (1, 64, 64), float32 in [0, 1]
```

Keep the bitmap as `uint8` through geometric transforms and convert it with
`ToDtype(torch.float32, scale=True)` near the model boundary. `ToImage()` does
not scale pixel values. `ToPureTensor()` removes the image subclass before the
payload enters a model. TorchVision remains an optional integration dependency;
TorchFont's renderer does not require it.

## Functional API

Deterministic operations are available from `torchfont.transforms.functional`:

```python
from torchfont import Outline
from torchfont.transforms import functional as F

outline = F.load_glyph(sample.ref)
outline = F.remove_overlaps(outline)
outline = F.quad_to_cubic(outline, merge_curves=True)
outline = F.affine(outline, angle=10.0)
bitmap = F.render_bitmap(outline, size=64)
shape = bitmap.shape
```

The functional API does not sample randomness. Random selection and parameter
sampling belong to the `Random*` transform classes.

### Single glyphs only

Every operation in this section accepts a single glyph. Passing a batched
`Outline` raises:

```python
F.affine(batch, angle=5.0)
# ValueError: affine operates on a single outline, got batch shape (64,);
#             iterate with unpad_outlines() first
```

Transforms run per sample, before collation. Batch a pipeline's output with
[`pad_outlines`](./core-types.md#pad-outlines) or a `DataLoader`.

### Differentiability

Gradient support varies by operation:

| Kernel | Differentiable |
| --- | --- |
| `affine` | yes |
| `coord_jitter` | yes, in both the outline and the noise |
| `horizontal_flip`, `vertical_flip` | only with `preserve_winding=False` |
| `quad_to_cubic`, `cubic_to_quad`, `merge_curves`, `split_segments` | no |
| `remove_overlaps`, `remove_overlap_groups` | no |
| `normalize_subpath_start_points`, `set_subpath_start_points`, `reorder_subpaths` | no |
| `render_bitmap` | no |

Passing an outline that requires grad to an operation marked "no" raises:

```python
outline.coords.requires_grad_()
F.remove_overlaps(outline)
# Raises RuntimeError
```

`affine` and the flips pivot around the tight bounding-box centre. Gradients flow
through the transformed coordinates but not through that centre.

### Devices

`LoadGlyph` returns CPU `float32` outlines. `Affine`, flip, curve, overlap, and
subpath transforms, as well as `RenderBitmap`, require CPU `float32` outlines.
Convert other outlines explicitly before calling them:

```python
outline = outline.to("cpu", torch.float32)
```

`RandomCoordJitter` and `functional.coord_jitter` preserve the input device and
floating point dtype.

### `torch.compile`

Functional pipelines can be used with `torch.compile`:

```python
import torch

# Required on PyTorch 2.5 when an operation can change the outline length.
torch._dynamo.config.capture_dynamic_output_shape_ops = True


def pipeline(types, coords):
    outline = Outline(types, coords)
    outline = F.remove_overlaps(outline)
    outline = F.cubic_to_quad(outline)
    outline = F.affine(outline, angle=10.0)
    return F.render_bitmap(outline, 32)


compiled = torch.compile(pipeline)
```
