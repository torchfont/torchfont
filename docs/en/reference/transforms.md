# Transforms

TorchFont transforms follow the torchvision v2 model: semantic data objects,
class-based stochastic transforms, deterministic functional kernels, and pytree
composition.

## Data types

```python
from torchfont import tf_tensors
from torchfont.structures import GlyphData, Outline
```

`Outline(types, coords)` keeps the two coupled tensors together.
`tf_tensors.TFTensor` is the semantic tensor base corresponding to torchvision's
`TVTensor`, and `tf_tensors.Bitmap(tensor)` is its rasterized-glyph subclass.
Ordinary tensor operations on a `TFTensor` return a plain tensor so semantic
dispatch does not leak into model code. In line with torchvision's `TVTensor`,
`clone()`, `detach()`, `pin_memory()`, `requires_grad_()`, and `to()` preserve the
subclass; convenience methods such as `float()` and `cpu()` follow the ordinary
operation rule. Use `tf_tensors.wrap(tensor, like=bitmap)` to restore a semantic
subclass without copying data. Custom `TFTensor` subclasses that carry metadata
can override the `wrap()` classmethod; copy-like operations and the public
`tf_tensors.wrap()` helper then preserve that metadata without a global registry.
`GlyphData[T]`
keeps a transformed payload, the original dataset sample, and the concrete
variation location together. Since its payload is generic, a pipeline can change
it from `Outline` to a bitmap without losing metadata.

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
`RandomLocation` performs the corresponding operation for
`VariableGlyphSample` and records the sampled location in `GlyphData.location`.

`Transform` flattens nested pytree inputs, samples parameters independently for
each matching semantic leaf, and restores the input structure. Wrap a transform
with `SameParams` when corresponding outlines, such as compatible instances of
one variable glyph, must receive the same flip, affine parameters, or
element-level random values. Random transforms use PyTorch's default RNG, so
`torch.manual_seed` and DataLoader worker seeding apply normally.
`SameParams(RandomLocation())` may likewise select one location for multiple
glyphs from the same font; sharing raw axis values across different fonts is
rejected.
Built-in transforms contain configuration only and remain pickle-friendly.
`Compose` registers its children in a `torch.nn.ModuleList`, including when it
is constructed from an ordinary list of modules. `RandomApply` and `SameParams`
register the single module they wrap. Plain callables are intentionally
unsupported: define a small `nn.Module` so its
behavior, representation, and pickle requirements remain explicit. This avoids
inheriting torchvision's historical unregistered callable-list behavior and
keeps module traversal, state dictionaries, hooks, and configuration display
consistent with PyTorch.

Containers invoke registered children through the module call path, so forward
hooks remain effective. `train()` and `eval()` propagate normally, but built-in
random transforms intentionally do not use the `training` flag: preprocessing
and model mode are separate concerns. Select a deterministic pipeline explicitly
for evaluation rather than relying on `eval()` to disable augmentation.

Like torchvision v2, transforms and containers accept either one pytree or
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
| Loading | `LoadGlyph`, `RandomLocation` |
| Containers | `Compose`, `RandomApply`, `SameParams` |
| Curves | `QuadToCubic`, `CubicToQuad`, `MergeCurves`, `RandomSplitSegments` |
| Outline | `RemoveOverlaps`, `RandomRemoveOverlaps` |
| Subpaths | `NormalizeSubpathStartPoints`, `RandomizeSubpathStartPoints`, `RandomizeSubpathOrder` |
| Geometry | `Affine`, `RandomAffine`, `HorizontalFlip`, `VerticalFlip`, `RandomHorizontalFlip`, `RandomVerticalFlip`, `RandomCoordJitter` |
| Output | `RenderBitmap`, `ToPureTensor` |

`RenderBitmap` changes each `Outline` leaf into a `Bitmap` containing a `uint8`
tensor. When these leaves are inside `GlyphData`, the sample and location remain
alongside the converted payload. `ToPureTensor` removes `TFTensor` subclasses
before data enters a model, without copying tensor storage.

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
