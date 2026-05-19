# DataLoader によるバッチ処理

<!-- markdownlint-disable MD013 -->

## なぜ DataLoader を使うのか

ニューラルネットワークの学習では、データを1件ずつ処理するのではなく、複数件をまとめたバッチ単位で処理します。バッチ処理により勾配の推定が安定し、GPU の並列演算を効率的に活用できます。`DataLoader` はバッチの構築・シャッフル・並列読み込みをまとめて担う PyTorch の標準ユーティリティです。

## `transform` を定義する

`GlyphSample` はグリフに関する情報をまとめて持っていますが、学習に必要なフィールドはタスクによって異なります。不要なフィールドを DataLoader に流すと転送コストが増えるため、`transform` で必要なテンソルだけを取り出します。

`GlyphDataset` には、PyTorch の Dataset と同様にアイテムごとに変換を適用する `transform` 引数があります。ここでは `GlyphSample` から `types` と `coords` を取り出す関数を定義し、Dataset に渡して動作を確認します。次のコードを実行してください。

```python
from torchfont.datasets import GlyphDataset, GlyphSample


def transform(sample: GlyphSample):
    return sample.types, sample.coords


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=transform,
)

types, coords = dataset[0]

print(types.shape)
print(coords.shape)
```

`transform` を渡すと、`dataset[0]` の返り値が `GlyphSample` から `(types, coords)` のタプルに変わります。`1` はこのグリフのシーケンス長で、グリフごとに異なります。実行すると次のような出力が得られます。

```
torch.Size([1])
torch.Size([1, 6])
```

## DataLoader を作成する

グリフのアウトライン系列は可変長のため、バッチ化には `collate_fn` が必要です。`pad_sequence` を使ってバッチ内のシーケンスを揃えます。次のコードを実行してください。

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample


def transform(sample: GlyphSample):
    return sample.types, sample.coords


def collate_fn(batch):
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=transform,
)

loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
types_t, coords_t = next(iter(loader))

print(types_t.shape)
print(coords_t.shape)
```

`collate_fn` はバッチ内の最長シーケンスに合わせて padding します。実行すると次のような出力が得られます。1 次元目はバッチサイズです。2 次元目はバッチ内の最長シーケンス長で、バッチごとに異なります。

```
torch.Size([64, 369])
torch.Size([64, 369, 6])
```

## マルチプロセス読み込み

`num_workers` と `prefetch_factor` を指定すると、データ読み込みをワーカープロセスで並列化できます。シーケンス長が長いと転送コストが大きくなるため、`transform` で先頭 512 要素に切り詰めます。`tqdm` で全バッチを読み込んでスループットを確認します。次のコードを実行してください。

```python
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset, GlyphSample


def transform(sample: GlyphSample):
    return sample.types[:512], sample.coords[:512]


def collate_fn(batch):
    types = pad_sequence([types for types, _ in batch], batch_first=True)
    coords = pad_sequence([coords for _, coords in batch], batch_first=True)
    return types, coords


dataset = GlyphDataset(
    root="data/google/fonts",
    patterns=(
        "apache/*/*.ttf",
        "ofl/*/*.ttf",
        "ufl/*/*.ttf",
        "!ofl/adobeblank/AdobeBlank-Regular.ttf",
    ),
    transform=transform,
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

実行すると次のような出力が得られます。`it/s` はバッチの処理速度です。1,246 万サンプルからなる Google Fonts 全体をわずか 2 分でイテレートできています。1 エポックがこの速度で回るため、実用的な学習ループに十分なスループットです。

```
len(dataset)=12460609
100%|██████████| 194698/194698 [02:03<00:00, 1570.64it/s]
```
