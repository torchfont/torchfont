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

## カスタム sample 形状

`collate_fn` は先頭のシーケンス次元だけを padding します。dataset transform が
末尾次元を増やす場合、その末尾次元は batch 化後も保持されます。
