# Transforms

TorchFont transforms follow the torchvision v2 model: semantic data objects,
class-based stochastic transforms, deterministic functional kernels, and pytree
composition.

## Data types

```python
from torchfont.transforms import Bitmap, GlyphData, Outline, OutlinePatches, TFTensor
```

`Outline(types, coords)` keeps the two coupled tensors together. `TFTensor` is
the semantic tensor base corresponding to torchvision's `TVTensor`, and
`Bitmap(tensor)` is its rasterized-glyph subclass. Ordinary tensor operations on
a `TFTensor` return a plain tensor so semantic dispatch does not leak into model
code; copying and device/dtype conversion preserve the subclass. `GlyphData[T]`
keeps a transformed payload, the original dataset sample, and the concrete
variation location together. Since its payload is generic, a pipeline can change
it from `Outline` to `OutlinePatches` or a bitmap without losing metadata.

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
        RandomApply([RandomSplitSegments(split_probability=0.2)], p=0.5),
        RemoveOverlaps(),
    ]
)

data = transform(sample)
outline = data.data
```

`LoadGlyph` accepts a `GlyphSample` or `GlyphRef`. A sample becomes
`GlyphData[Outline]`, while a bare reference becomes `Outline`.
`RandomLocation` performs the corresponding operation for
`VariableGlyphSample` and records the sampled location in `GlyphData.location`.

`Transform` flattens nested pytree inputs, calls `make_params()` once, applies
the resulting parameters to every matching semantic leaf, and restores the
input structure. Thus corresponding outlines receive the same flip, affine
parameters, or element-level random values. Random transforms use PyTorch's
default RNG, so `torch.manual_seed` and DataLoader worker seeding apply normally.
Built-in transforms contain configuration only and remain pickle-friendly.
Custom callables passed to a container are pickle-friendly only if the callable is.
As in torchvision, callable sequences in `Compose` and `RandomApply` are not
registered as trainable child modules. Transform pipelines should therefore hold
configuration, not parameters or mutable runtime state.

Like torchvision v2, transforms and containers accept either one pytree or
multiple positional inputs. `check_inputs()` is available to custom transforms
that need to check relationships between leaves before sampling parameters.
`Compose` and `RandomApply` require a non-empty sequence of callables.
Parameters are shared across matching leaves in one call. Call a transform
separately for independent samples that should receive independent randomness.

`RandomApply(transforms, p)` controls whether a complete transform sequence is
applied. Probabilities such as `RandomSplitSegments.split_probability` control
behavior inside an already-applied transform.

## Built-in transforms

| Category | Transforms |
| --- | --- |
| Loading | `LoadGlyph`, `RandomLocation` |
| Containers | `Compose`, `RandomApply` |
| Curves | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| Outline | `RemoveOverlaps`, `RandomRemoveOverlaps`, `Patchify` |
| Subpaths | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| Geometry | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| Output | `RenderBitmap` |

`Patchify` changes each `Outline` leaf into `OutlinePatches`. `RenderBitmap`
changes it into a `Bitmap` containing a `uint8` tensor. When these leaves are inside `GlyphData`, the
sample and location remain alongside the converted payload.

## Functional kernels

Deterministic operations are available from `torchfont.transforms.functional`:

```python
from torchfont.transforms import Outline
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

Functionals dispatch kernels by the semantic input type. Downstream outline
subclasses can specialize a functional without changing the transform class:

```python
@F.register_kernel(F.horizontal_flip, CustomOutline)
def horizontal_flip_custom(inpt, *, preserve_winding=True): ...
```

Built-in outline transforms select `Outline` instances, so registered kernels
extend dispatch for `Outline` subclasses. A custom semantic type also needs a
custom transform whose `_transformed_types` selects that type. Override
`_needs_transform_list()` only when selection depends on relationships between
multiple leaves.
