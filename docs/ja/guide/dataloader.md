# DataLoader によるバッチ処理

<!-- markdownlint-disable MD013 -->

## なぜ DataLoader を使うのか

ニューラルネットワークの学習では、データを 1 件ずつ処理するのではなく、複数件をまとめたバッチ単位で処理します。バッチ処理により勾配の推定が安定し、GPU の並列演算を効率的に活用できます。`DataLoader` はバッチの構築・シャッフル・並列読み込みをまとめて担う PyTorch の標準ユーティリティです。

## `transform` を定義する

`GlyphSample` はグリフ参照とターゲットインデックスを持ちます。パイプラインの最初に
`LoadGlyph` を使うと、サンプルのメタデータを保持したまま意味型 `Outline` を読み込めます。

`GlyphDataset` には、PyTorch のデータセットと同様にアイテムごとに変換を適用する
`transform` 引数があります。`LoadGlyph()` を直接渡して動作を確認します。

```python
from torchfont.datasets import GlyphDataset
from torchfont import GlyphData, Outline
from torchfont.transforms import LoadGlyph


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=LoadGlyph(),
)

data: GlyphData[Outline] = dataset[0]

print(data.data.shape)
print(data.data.coords.shape)
```

`LoadGlyph` を渡すと、`dataset[0]` は `GlyphData[Outline]` を返します。`data` Field が
Outline で、ほかの Field が参照、Location、Target を保持します。最初の形状は `(N,)`、
次は `(N, 6)` で、`N` は Glyph ごとに異なります。例えば次のようになります。

```
torch.Size([37])
torch.Size([37, 6])
```

## DataLoader を作成する

グリフのアウトライン系列は可変長なので、モデルの入力契約に合うローカルな
`collate_fn` を定義します。Payload には `pad_outlines` を使い、モデルが必要とする
Target だけを Tensor に変換します。

```python
import math

import torch
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont import GlyphData, Outline, pad_outlines
from torchfont.transforms import LoadGlyph


def collate_fn(samples: list[GlyphData[Outline]]):
    return {
        "outline": pad_outlines([sample.data for sample in samples]),
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


dataset = GlyphDataset(
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

print(batch["outline"].shape)
print(batch["outline"].coords.shape)
print(batch["weight"].shape)
```

Outline はバッチ内の最長のものに合わせてパディングされます。1 次元目はバッチサイズです。2 次元目はバッチ内の最長シーケンス長で、バッチごとに異なります。Target は長さ `batch_size` の 1 次元テンソルになります。次のような出力が得られます。

```
torch.Size([64, 369])
torch.Size([64, 369, 6])
torch.Size([64])
```

## パディング済みバッチを扱う

パディングされた要素は `ElementType.PAD` です。その値と比較して復元するのではなく、`padding_mask` を読んでください。これは Attention モジュールが `key_padding_mask` として要求するものそのままです。

```python
mask = batch["outline"].padding_mask  # (64, 369)、パディング位置が True
```

`unpad_outlines()` は Padding 済み Batch を明示的に分割し、入力した単一 Outline に戻します。一方、`Outline.unbind()` は通常の Tensor 操作と同じく Padding を保持します。

```python
from torchfont import unpad_outlines

singles = unpad_outlines(batch["outline"])

print(len(singles), singles[0].shape)
```

`torchfont.nn` のモジュールはバッチ化の有無を問わず `Outline` を受け取るので、パディング済みバッチをそのままモデルに渡せます。

```python
from torchfont.nn import OutlineEmbedding

tokens = OutlineEmbedding(embedding_dim=256)(batch["outline"])

print(tokens.shape)  # (64, 369, 256)
```

## DataLoader を使わずにバッチ化する

`pad_outlines` は同じパディング処理を直接呼び出せます。

```python
from torchfont import pad_outlines

batched = pad_outlines([dataset[0].data, dataset[1].data])

print(batched.shape)
```

## マルチプロセス読み込み

`num_workers` と `prefetch_factor` を指定すると、データ読み込みをワーカープロセスで並列化できます。

各バッチはその中の最長 Outline に合わせてパディングされるため、巨大なグリフが 1 つ混ざるとバッチ全体、ひいては学習プロセスへの転送量が膨らみます。次の例ではローカルな `collate_fn` で各 Outline を 512 要素に打ち切ります。ワーカープロセスは `collate_fn` を pickle するため、lambda ではなくモジュールレベルの関数として定義してください。

`tqdm` で全バッチを読み込んでスループットを確認します。次のコードを実行してください。

```python
import torch
from tqdm import tqdm
from torch.utils.data import DataLoader

from torchfont import GlyphData, Outline, pad_outlines
from torchfont.datasets import GlyphDataset
from torchfont.transforms import LoadGlyph

MAX_ELEMENTS = 512


def collate_fn(samples: list[GlyphData[Outline]]):
    return {
        "outline": pad_outlines([sample.data[:MAX_ELEMENTS] for sample in samples]),
        "font_idx": torch.tensor(
            [sample.font_idx for sample in samples], dtype=torch.long
        ),
    }


dataset = GlyphDataset(
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
    num_workers=8,
    prefetch_factor=2,
)

print(f"{len(dataset)=}")

for batch in tqdm(loader):
    pass
```

データセット長は、選択したフォントファイルと各フォントの文字マップによって変わります。プログレスバーの `it/s` はバッチの処理速度です。ストレージや学習環境に適したワーカー数とプリフェッチ設定を決める指標として利用してください。

```
len(dataset)=...
100%|██████████| .../... [..., ...it/s]
```

::: tip 打ち切らずにパディングを抑える
打ち切りは幾何情報を捨てます。Outline 全体を保ったままパディングのコストを避けるには、長さを打ち切るのではなく、長さを考慮した `Sampler` で近い長さの Glyph をまとめてください。
:::
