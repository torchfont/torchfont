# DataLoader によるバッチ処理

<!-- markdownlint-disable MD013 -->

## なぜ DataLoader を使うのか

ニューラルネットワークの学習では、複数のサンプルをバッチにまとめて処理するのが一般的です。
`DataLoader` は、バッチの構築、シャッフル、並列読み込みを担う PyTorch の標準機能です。

## `transform` を定義する

`CodepointSample` はグリフ参照とターゲットインデックスを持ちます。パイプラインの最初に
`LoadGlyph` を使うと、サンプルのメタデータを保持したまま意味型 `Outline` を読み込めます。

`CodepointDataset` には、PyTorch のデータセットと同様にアイテムごとに変換を適用する
`transform` 引数があります。`LoadGlyph()` を直接渡して動作を確認します。

```python
from torchfont.datasets import CodepointDataset
from torchfont import CodepointData, Outline
from torchfont.transforms import LoadGlyph


dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

data: CodepointData[Outline] = dataset[0]

print(data.data.shape)
print(data.data.coords.shape)
```

`LoadGlyph` を渡すと、`dataset[0]` は `CodepointData[Outline]` を返します。`data` フィールドが
`Outline` で、ほかのフィールドが参照、バリエーション位置、ターゲットを保持します。
最初の shape は `(N,)`、次は `(N, 6)` で、`N` はグリフごとに異なります。
例えば次のようになります。

```
torch.Size([37])
torch.Size([37, 6])
```

## DataLoader を作成する

グリフのアウトライン系列は可変長なので、モデルの入力契約に合うローカルな
`collate_fn` を定義します。アウトライン Tensor には PyTorch の `pad_sequence` を使い、
モデルが必要とするターゲットだけをテンソルに変換します。

```python
import math

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import CodepointDataset
from torchfont import CodepointData, ElementType, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(samples: list[CodepointData[Outline]]):
    outlines = [sample.data for sample in samples]
    return {
        "types": pad_sequence(
            [outline.types for outline in outlines],
            batch_first=True,
            padding_value=ElementType.PAD,
        ),
        "coords": pad_sequence(
            [outline.coords for outline in outlines], batch_first=True
        ),
        "lengths": torch.tensor([len(outline) for outline in outlines]),
        "font_idx": torch.tensor(
            [sample.font_idx for sample in samples], dtype=torch.long
        ),
        "weight": torch.tensor(
            [
                math.nan if sample.weight is None else sample.weight
                for sample in samples
            ],
            dtype=torch.float32,
        ),
    }


dataset = CodepointDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

loader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    collate_fn=collate_fn,
)
batch = next(iter(loader))

print(batch["types"].shape)
print(batch["coords"].shape)
print(batch["weight"].shape)
```

`Outline` はバッチ内の最長のものに合わせてパディングされます。1 次元目はバッチサイズ、
2 次元目はバッチごとに異なる最長シーケンス長です。ターゲットは長さ `batch_size` の
1 次元テンソルになります。次のような出力が得られます。

```
torch.Size([64, 369])
torch.Size([64, 369, 6])
torch.Size([64])
```

## パディング済みバッチを扱う

パディングされた要素は `ElementType.PAD` です。Attention モジュールが
`key_padding_mask` として期待する Boolean Mask を Element Type から直接作ります。

```python
mask = batch["types"] == ElementType.PAD  # (64, 369)、パディング位置が True
```

Padding 済み Tensor を復元する必要がある場合は元の Length を保持し、PyTorch の
`unpad_sequence` を使います。

```python
from torch.nn.utils.rnn import unpad_sequence

types = unpad_sequence(batch["types"], batch["lengths"], batch_first=True)
coords = unpad_sequence(batch["coords"], batch["lengths"], batch_first=True)
```

`torchfont.nn` のモジュールには Padding 済み Tensor を直接渡します。

```python
from torchfont.nn import OutlineEmbedding

tokens = OutlineEmbedding(embedding_dim=256)(batch["types"], batch["coords"])

print(tokens.shape)  # (64, 369, 256)
```
