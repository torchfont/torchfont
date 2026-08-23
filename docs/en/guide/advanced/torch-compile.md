# Compiling Transform Pipelines

TorchFont functional transforms can be captured with
[`torch.compile`](https://docs.pytorch.org/docs/stable/generated/torch.compile.html).
Load the glyph first, then compile the tensor-only part of the pipeline:

```python
import torch

from torchfont import Outline
from torchfont.datasets import GlyphDataset
from torchfont.transforms import functional as F

dataset = GlyphDataset("data/fonts", codepoints=[ord("A")])
outline = F.load_glyph(dataset[0].ref)


def pipeline(types: torch.Tensor, coords: torch.Tensor) -> torch.Tensor:
    outline = Outline(types, coords)
    outline = F.remove_overlaps(outline)
    outline = F.cubic_to_quad(outline)
    outline = F.affine(outline, angle=10.0, scale=1.1)
    outline = F.normalize_subpath_start_points(outline)
    return F.render_bitmap(outline, size=32)


# Required by PyTorch 2.5 for operations with data-dependent output shapes.
torch._dynamo.config.capture_dynamic_output_shape_ops = True

compiled_pipeline = torch.compile(pipeline, fullgraph=True)
bitmap = compiled_pipeline(outline.types, outline.coords)

print(bitmap.shape)  # torch.Size([32, 32])
print(bitmap.dtype)  # torch.uint8
```

`fullgraph=True` makes compilation fail when the function contains a graph
break, which is useful for checking that the complete transform pipeline was
captured. TorchFont's native operations remain CPU custom operations; compiling
the pipeline captures the surrounding PyTorch graph but does not fuse the
implementation inside those operations.

## Dynamic outline lengths

Operations such as `remove_overlaps` and `cubic_to_quad` can change the number
of outline elements. On PyTorch 2.5, enable capture of
data-dependent output shapes before compiling such a pipeline:

```python
torch._dynamo.config.capture_dynamic_output_shape_ops = True
```

Leave the `dynamic` argument at its default first. PyTorch can recompile with a
more dynamic graph when input lengths vary. Use `dynamic=True` only when you
know that outlines with many different lengths will be passed to the same
compiled function.

## Constraints

- Load fonts and select variable-font locations outside the compiled function.
- Outline topology and rendering operations require CPU `float32` coordinates
  and `torch.long` types.
- Topology-changing operations and `render_bitmap` are not differentiable.
  `torch.compile` does not change their autograd support.
- Use functional transforms with explicit arguments inside the compiled
  function. Keep Python-side parameter sampling outside the compiled region.
