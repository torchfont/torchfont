# Neural Network Building Blocks

TorchFont provides small font-specific building blocks in `torchfont.nn` that
accept element-type and coordinate tensors.

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

tokens = embedding(types, coords)  # (1, 4, 256)
padding_mask = types == ElementType.PAD  # (1, 4)
```

The module adds a learned element-type embedding to a bias-free linear
projection of the six continuous coordinates. Coordinate components inactive
for an element type are masked before projection. `ElementType.PAD` is the
`padding_idx`, and complete padding tokens produce zero vectors. Add positional
encoding appropriate for the downstream sequence architecture separately.

The constructor accepts PyTorch-style `device` and `dtype` keyword arguments:

```python
embedding = OutlineEmbedding(256, device="cuda", dtype=torch.float16)
```

## `OutlineLoss`

`OutlineLoss` combines element-type classification and coordinate regression
into one training objective:

```python
from torchfont.nn import OutlineLoss

criterion = OutlineLoss(
    type_weight=1.0,
    coordinate_weight=0.5,
    reduction="mean",
)
loss = criterion(type_logits, predicted_coords, target_types, target_coords)
```

`type_logits` has shape `(..., N, len(ElementType))`. Cross entropy and
coordinate error (MSE by default) are combined using `type_weight` and `coordinate_weight`.
Padding elements and inactive coordinate slots do not contribute. An
all-padding input produces a differentiable zero loss.

`reduction` supports `"none"`, `"mean"`, and `"sum"`. With `"none"`, losses
are summed within each outline and the output shape is `target_types.shape[:-1]`.
With `"sum"`, these outline losses are summed. With `"mean"` (the default),
cross entropy is averaged over all non-padding elements and coordinate error is
averaged independently over all active coordinate scalars.

The equivalent functional API is `torchfont.nn.functional.outline_loss`.
