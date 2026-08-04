# Neural Network Building Blocks

TorchFont provides small font-specific building blocks in `torchfont.nn`.
They operate on ordinary tensors so they compose with standard PyTorch modules,
devices, dtypes, autograd, and serialization.

## `OutlineEmbedding`

`OutlineEmbedding` converts padded outline element types and coordinates into
token features:

```python
import torch

from torchfont import ElementType
from torchfont.nn import OutlineEmbedding

embedding = OutlineEmbedding(embedding_dim=256)
types = torch.tensor(
    [[ElementType.MOVE_TO, ElementType.LINE_TO, ElementType.END, ElementType.PAD]]
)
coords = torch.zeros((1, 4, 6))  # (batch, sequence, 6)
tokens = embedding(types, coords)  # (batch, sequence, 256)
padding_mask = types == ElementType.PAD
```

The module adds a learned element-type embedding to a bias-free linear
projection of the six continuous coordinates. Coordinate components inactive
for an element type are masked before projection. `ElementType.PAD` is the
`padding_idx`, and complete padding tokens produce zero vectors. Positional
information is intentionally separate: use the positional encoding appropriate
for the downstream sequence architecture.

The constructor accepts PyTorch-style `device` and `dtype` keyword arguments:

```python
embedding = OutlineEmbedding(256, device="cuda", dtype=torch.float16)
```

## `coordinate_loss`

`coordinate_loss` computes squared error only over coordinates that carry
meaning for the target element type:

```python
from torchfont.nn import functional as F

loss = F.coordinate_loss(predicted_coords, target_coords, target_types)
```

- `MOVE_TO` and `LINE_TO`: endpoint only
- `QUAD_TO`: first control point and endpoint
- `CURVE_TO`: both control points and endpoint
- `CLOSE`, `END`, and `PAD`: no coordinates

Inputs and targets have shape `(..., N, 6)`, while element types have shape
`(..., N)`. Supported reductions are `"none"`, `"mean"`, and `"sum"`.
The mean is taken over active coordinate scalars rather than padded storage.
If there are no active coordinates, the mean is a differentiable zero.
Non-finite values in inactive coordinate slots are ignored.

## `OutlineLoss`

`OutlineLoss` combines element-type classification and coordinate regression
into one training objective:

```python
from torchfont.nn import OutlineLoss

criterion = OutlineLoss(type_weight=1.0, coordinate_weight=0.5)
loss = criterion(
    type_logits,
    predicted_coords,
    target_types,
    target_coords,
)
```

`type_logits` has shape `(..., N, TYPE_DIM)`. The other tensors have the same
shapes accepted by `coordinate_loss`. Cross entropy is averaged over non-padding
elements, while coordinate error is averaged independently over active
coordinate scalars. The two means are then combined using `type_weight` and
`coordinate_weight`. This keeps their relative weighting stable across outline
lengths and padding amounts. An all-padding input produces a differentiable
zero loss.

The loss intentionally has no `reduction` argument. Use PyTorch cross entropy
and `coordinate_loss` separately when unreduced components or a different
aggregation are required. The equivalent functional API is
`torchfont.nn.functional.outline_loss`.
