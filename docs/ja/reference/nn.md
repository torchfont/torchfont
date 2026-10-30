# ニューラルネットワーク構成要素

TorchFont は、要素型と座標の Tensor を受け取るフォント固有の構成要素を
`torchfont.nn` で提供します。

## `OutlineEmbedding`

`OutlineEmbedding` は、Padding 済みのアウトライン要素型と座標を Token Feature に変換します。

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

学習可能な要素型 Embedding と、Bias なしの線形層による6次元連続座標の射影を加算します。
要素型に対して無効な座標成分は、射影前に Mask されます。
`ElementType.PAD` を `padding_idx` とし、Padding Token 全体の出力はゼロになります。
下流の系列アーキテクチャに適した位置 Encoding は別途追加します。

Constructor は PyTorch と同様に `device` と `dtype` Keyword Argument を受け取ります。

```python
embedding = OutlineEmbedding(256, device="cuda", dtype=torch.float16)
```

## `OutlineLoss`

`OutlineLoss` は、要素型の分類と座標の回帰を一つの学習目的に統合します。

```python
from torchfont.nn import OutlineLoss

criterion = OutlineLoss(
    type_weight=1.0,
    coordinate_weight=0.5,
    reduction="mean",
)
loss = criterion(type_logits, predicted_coords, target_types, target_coords)
```

`type_logits` の Shape は `(..., N, len(ElementType))` です。Cross Entropy と座標誤差（既定は MSE）を
`type_weight` と `coordinate_weight` で重み付けします。Padding 要素と無効な座標 Slot は
Loss に寄与しません。すべてが Padding の Input に対しては、微分可能なゼロ Loss を返します。

`reduction` は `"none"`、`"mean"`、`"sum"` に対応します。`"none"` では Outline ごとに
Loss を合計し、`target_types.shape[:-1]` Shape の Tensor を返します。`"sum"` ではそれらを
さらに合計します。既定の `"mean"` では、Cross Entropy をすべての Padding 以外の要素で、
座標誤差をすべての有効な座標 Scalar で独立に平均します。

同等の関数 API は `torchfont.nn.functional.outline_loss` です。
