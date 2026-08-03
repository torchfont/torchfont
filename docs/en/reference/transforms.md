# Transforms

TorchFont transforms follow the TorchVision Transforms v2
(`torchvision.transforms.v2`) model: semantic data objects,
class-based stochastic transforms, deterministic functional kernels, and pytree
composition.

## Data types

```python
from torchfont.structures import GlyphData, Outline
```

`Outline(types, coords)` keeps the two coupled tensors together.
`GlyphData[T]` keeps a transformed payload, the original dataset sample, and the concrete
variation location together. Since its payload is generic, a pipeline can change
it from `Outline` to a plain bitmap tensor without losing metadata. Rasterized
glyphs do not need a TorchFont-specific tensor subclass: they cross into image
semantics explicitly through TorchVision's `ToImage()` when required.

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

`LoadGlyph` accepts a `GlyphSample` or `GlyphRef`. A sample becomes
`GlyphData[Outline]`, while a bare reference becomes `Outline`.
`LoadGlyph` uses the face's default location unless `location="random"` is set.
The random policy samples one location for a `GlyphSample` or `GlyphRef` and
records it in `GlyphData.location`; on a static face it naturally uses an empty
location.

`Transform` flattens nested pytree inputs, samples parameters independently for
each matching semantic leaf, and restores the input structure. Wrap a transform
with `SameParams` when corresponding outlines, such as multiple glyphs from the
same variable font at one location, must receive the same flip, affine parameters, or
element-level random values. Random transforms use PyTorch's default RNG, so
`torch.manual_seed` and DataLoader worker seeding apply normally.
`SameParams(LoadGlyph(location="random"))` may likewise select one location for
multiple glyphs from the same font; sharing raw axis values across different
fonts is rejected.
Built-in transforms contain configuration only and remain pickle-friendly.
`Compose` registers its children in a `torch.nn.ModuleList`, including when it
is constructed from an ordinary list of modules. `RandomApply` and `SameParams`
register the single module they wrap. Plain callables are intentionally
unsupported: define a small `nn.Module` so its
behavior, representation, and pickle requirements remain explicit. This keeps
module traversal, state dictionaries, hooks, and configuration display
consistent with PyTorch.

Containers invoke registered children through the module call path, so forward
hooks remain effective. `train()` and `eval()` propagate normally, but built-in
random transforms intentionally do not use the `training` flag: preprocessing
and model mode are separate concerns. Select a deterministic pipeline explicitly
for evaluation rather than relying on `eval()` to disable augmentation.

Like `torchvision.transforms.v2`, transforms and containers accept either one pytree or
multiple positional inputs. `check_inputs()` is available to custom transforms
that need to check relationships between leaves before sampling parameters.
`Compose` immediately materializes an iterable of `nn.Module` objects into an
`nn.ModuleList`; an empty iterable is an identity transform. `RandomApply` wraps
one `nn.Module`, while `SameParams` wraps one `Transform`. Use `Compose` inside
`RandomApply` when grouping several transforms.

`RandomApply(transform, p)` controls whether one transform is applied.
Probabilities such as `RandomSplitSegments.split_probability` control behavior
inside an already-applied transform.

## Built-in transforms

| Category | Transforms |
| --- | --- |
| Loading | `LoadGlyph` |
| Containers | `Compose`, `RandomApply`, `SameParams` |
| Curves | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| Outline | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| Subpaths | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| Geometry | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| Output | `RenderBitmap` |

`RenderBitmap` changes each `Outline` leaf into a plain `uint8` tensor. When
these leaves are inside `GlyphData`, the sample and location remain alongside
the converted payload.

### Using rendered glyphs with TorchVision

`RenderBitmap` returns a plain grayscale `H x W` tensor. Use
`torchvision.transforms.v2.ToImage()` as the boundary into an image pipeline:
it adds the channel
dimension and returns a `tv_tensors.Image` of shape `1 x H x W`. It preserves
the enclosing `GlyphData` because both libraries operate on pytrees.

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

## Functional kernels

Deterministic operations are available from `torchfont.transforms.functional`:

```python
from torchfont.structures import Outline
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

These transforms are `nn.Module` objects for composition, registration, PyTorch
RNG behavior, and pytree processing. Rust-backed outline functionals cross a
CPU/NumPy boundary and are preprocessing operations: they are not promised to
participate in autograd, remain on an accelerator, or be captured by
`torch.compile`. Functionals currently operate on the single semantic input type
documented by their signatures; TorchFont does not add a kernel registry until
multiple outline representations require one.
