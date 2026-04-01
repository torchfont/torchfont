# DataLoader との統合

<!-- markdownlint-disable MD013 -->

TorchFont の Dataset は `torch.utils.data.Dataset` を継承しているため、通常の PyTorch ワークフローで使えます。

## まずは最小確認（`batch_size=1`）

```python
from torch.utils.data import DataLoader
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(root="~/fonts")
loader = DataLoader(dataset, batch_size=1, shuffle=True)

sample = next(iter(loader))
print(sample.types.shape, sample.coords.shape)  # (1, seq_len), (1, seq_len, 6)
print(sample.style_idx.shape, sample.content_idx.shape)  # (1,), (1,)
```

この例は動作確認用です。`batch_size > 1` では可変長シーケンスを扱うため、通常は padding 対応の `collate_fn` が必要です。

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
print(batch.mask.shape)
```

`num_workers > 0` のときだけ、プリフェッチと multiprocessing の設定を有効にします。`num_workers=0` なら、これらの引数は指定しないでください。

|OS|推奨 `multiprocessing_context`|
|---|---|
|Linux|`"fork"`|
|macOS|`"spawn"` または `"forkserver"`|
|Windows|`"spawn"`|

## パディングマスク

組み込みの `collate_fn` は `GlyphBatch.mask` を返します。`True` が有効な
シーケンス位置です。

```python
valid_mask = batch.mask
padding_mask = ~batch.mask
```

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
