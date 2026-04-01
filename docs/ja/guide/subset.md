# サブセット抽出

TorchFont では `dataset.targets` を使うと、スタイル/文字ラベルに基づくサブセットを簡単に作れます。

## 前提

以下の例では構築済み Dataset を使います。すぐ試す場合は次のように準備できます。

```python
import torch
from torch.utils.data import Subset
from torchfont.datasets import GlyphDataset

dataset = GlyphDataset(
    root="tests/fonts",
    patterns=("*.ttf",),
    codepoint_filter=range(0x80),
)

t = dataset.targets
if t.numel() == 0:
    raise ValueError("dataset が空です。patterns/codepoint_filter を緩めてください")
print(t.shape)  # (N, 2)
```

checkout 済みのフォントリポジトリを使う場合も、変えるのは `root` と
`patterns` だけです。

- `t[:, 0]`: `style_idx`
- `t[:, 1]`: `content_idx`

## 実在するラベルを先に選ぶ

観測済みサンプル由来の index を使うと、複合条件の例が空集合になりにくくなります。

```python
style_idx = t[0, 0].item()
content_idx = t[0, 1].item()
```

## よく使う抽出パターン

```python
# スタイルで抽出
style_indices = torch.where(t[:, 0] == style_idx)[0].tolist()
style_subset = Subset(dataset, style_indices)

# 文字で抽出
content_indices = torch.where(t[:, 1] == content_idx)[0].tolist()
content_subset = Subset(dataset, content_indices)

# 複合条件で抽出
mask = (t[:, 0] == style_idx) & (t[:, 1] == content_idx)
combined_indices = torch.where(mask)[0].tolist()
subset = Subset(dataset, combined_indices)
```

## 名前ベースで抽出する

`style_classes` / `content_classes` を使うと、人間が読める名前で指定できます。

```python
style_name = dataset.style_classes[style_idx]
style_indices = [
    idx for idx, name in enumerate(dataset.style_classes) if name == style_name
]
style_tensor = torch.as_tensor(style_indices, dtype=torch.long)
style_mask = torch.isin(t[:, 0], style_tensor)

char = dataset.content_classes[content_idx]
content_idx_from_name = dataset.content_class_to_idx[char]

mask = style_mask & (t[:, 1] == content_idx_from_name)
indices = torch.where(mask)[0].tolist()
subset = Subset(dataset, indices)
```

## 完全な実行例

実行可能なスクリプト名は `examples/subset_by_targets.py` です。サンプル全体は [サンプル集](/ja/examples/) を参照してください。
