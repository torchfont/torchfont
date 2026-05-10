# Google Fonts をローカル checkout から使う

Google Fonts リポジトリをローカルに checkout し、そのディレクトリを
`GlyphDataset` で開くのが基本です。

## 最小例

```bash
git clone --depth 1 https://github.com/google/fonts data/google/fonts
```

```python
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
)

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")
```

## この流れを使う理由

- リポジトリ同期は通常の Git ツール側に寄せられる
- Google Fonts も他のコーパスと同じ `GlyphDataset(root=...)` で扱える
- 再現性は checkout の commit を外側で記録すればよい

## 推奨 `patterns`

次の include / exclude は、`GlyphDataset` を直接使うときの既定候補として
扱いやすいです。

```python
(
    "apache/*/*.ttf",
    "ofl/*/*.ttf",
    "ufl/*/*.ttf",
    "!ofl/adobeblank/AdobeBlank-Regular.ttf",
)
```

必要な部分だけに絞りたい場合は、`patterns` をより狭くしてください。

### 文字集合を絞る

```python
dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=("ofl/*/*.ttf",),
    codepoints=range(0x30, 0x3A),  # 0-9
)
```

`codepoints` に含まれない文字はインデックス時に除外されます。

## 更新フロー

checkout は Git 側で更新し、そのあと Dataset を作り直してください。

```bash
git -C data/google/fonts pull --ff-only
git -C data/google/fonts rev-parse HEAD
```

厳密な再現性が必要なら、この commit hash を別途保存します。

## 学習パイプライン例

```python
import sys

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample
from torchfont.transforms import patchify, quad_to_cubic, render_bitmap


def transform(sample: GlyphSample) -> tuple[Tensor, Tensor, Tensor]:
    types = sample.types[:512]
    coords = sample.coords[:512]
    types, coords = quad_to_cubic(types, coords)
    bitmap = render_bitmap(types, coords)
    patch_types, patch_coords = patchify(types, coords, patch_size=32)
    return patch_types, patch_coords, bitmap


def collate_fn(
    batch: list[tuple[Tensor, Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    types = pad_sequence([types for types, _, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords, _ in batch], batch_first=True)
    bitmaps = torch.stack([bitmap for _, _, bitmap in batch])
    return types, coords, bitmaps


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=("ofl/*/*.ttf",),
    transform=transform,
)

num_workers = 8
loader_kwargs = {
    "batch_size": 64,
    "shuffle": True,
    "num_workers": num_workers,
    "collate_fn": collate_fn,
}

if num_workers > 0:
    loader_kwargs["prefetch_factor"] = 2
    loader_kwargs["multiprocessing_context"] = (
        "fork" if sys.platform.startswith("linux") else "spawn"
    )

loader = DataLoader(dataset, **loader_kwargs)
```

::: tip プラットフォーム別 multiprocessing

- Linux: `"fork"` が高速なことが多い
- macOS: `"spawn"` または `"forkserver"`
- Windows: `"spawn"`
- `num_workers=0` の場合は `prefetch_factor` と
  `multiprocessing_context` を外す

:::

## 関連ページ

- [Git リポジトリ](/ja/guide/git-repos)
- [DataLoader との統合](/ja/guide/dataloader)
- [サブセット抽出](/ja/guide/subset)
