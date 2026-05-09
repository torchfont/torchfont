# Transform Utilities

`torchfont.transforms` は `types` / `coords` tensor に対する stateless な
utility 関数を提供します。固定の pipeline 抽象は用意せず、
`GlyphDataset(transform=...)` に渡す callable や training step 側の前処理を、
用途に合わせて自由に実装する形を基本にします。

## `limit_sequence_length`

```python
from torchfont.transforms import limit_sequence_length

types, coords = limit_sequence_length(types, coords, max_len=512)
```

`max_len` まで切り詰めた `(types, coords)` を返します。
`max_len` は `>= 0` である必要があります。

## `quad_to_cubic`

```python
from torchfont.transforms import quad_to_cubic

types, coords = quad_to_cubic(types, coords)
```

`CommandType.QUAD_TO` を `CommandType.CURVE_TO` に変換します。command と
coordinate の shape は変わりません。

## `patchify`

```python
from torchfont.transforms import patchify

types, coords = patchify(types, coords, patch_size=32)
```

sequence 末尾をゼロ padding して `patch_size` に揃え、固定長 patch に変形します。

- input: `types=(seq_len,)`, `coords=(seq_len, 6)`
- output: `types=(num_patches, patch_size)`,
  `coords=(num_patches, patch_size, 6)`

`patch_size` は `>= 1` である必要があります。

## 例

```python
import dataclasses

from torchfont.transforms import limit_sequence_length, patchify, quad_to_cubic


def transform(sample):
    types, coords = quad_to_cubic(sample.types, sample.coords)
    types, coords = limit_sequence_length(types, coords, max_len=512)
    types, coords = patchify(types, coords, patch_size=32)
    return dataclasses.replace(sample, types=types, coords=coords)
```
