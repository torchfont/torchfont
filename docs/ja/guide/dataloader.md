# DataLoader との統合

<!-- markdownlint-disable MD013 -->

TorchFont の Dataset は `torch.utils.data.Dataset` を継承しているため、通常の PyTorch ワークフローで使えます。

## まずは最小確認（`batch_size=1`）

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="~/fonts")
sample = dataset[0]
print(sample.types.shape, sample.coords.shape)  # (seq_len,), (seq_len, 6)
print(sample.style_idx, sample.content_idx)
```

この例は動作確認用です。バッチ化には `collate_fn` を使ってください。

## 学習向けの `collate_fn`

```python
import sys

from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn


dataset = GlyphDataset(root="~/fonts")
num_workers = 8
mp_context = "fork" if sys.platform.startswith("linux") else "spawn"

loader_kwargs = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": num_workers,
    "collate_fn": collate_fn,
}

if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
    loader_kwargs["multiprocessing_context"] = mp_context

loader = DataLoader(dataset, **loader_kwargs)
batch = next(iter(loader))

print(batch.types.shape)
print(batch.coords.shape)
print(batch.targets.shape)
print(batch.metrics.shape)
```

`num_workers > 0` のときだけ、プリフェッチと multiprocessing の設定を有効にします。`num_workers=0` なら、これらの引数は指定しないでください。

|OS|推奨 `multiprocessing_context`|
|---|---|
|Linux|`"fork"`|
|macOS|`"spawn"` または `"forkserver"`|
|Windows|`"spawn"`|

## `Patchify` を使う場合

`Patchify` で固定長パッチへ分割しておくと、バッチ時の扱いが単純になります。

```python
from torchfont.transforms import Compose, LimitSequenceLength, Patchify

transform = Compose([
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])
```

この場合、`types.shape` は `(num_patches, 32)` になります。サンプルごとに
`num_patches` が異なる可能性は残るため、サンプル単位でバッチ化するなら
`collate_fn` 側で padding が必要になることがあります。
