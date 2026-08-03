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
from torchfont.structures import GlyphData, Outline
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
types, coords = data.data.types, data.data.coords

print(types.shape)
print(coords.shape)
```

`LoadGlyph` を渡すと、`dataset[0]` は `GlyphData[Outline]` を返します。`data`
フィールドがアウトライン、`sample` フィールドが元のメタデータです。最初の形状は `(N,)`、
次は `(N, 6)` で、`N` はグリフごとに異なります。例えば次のようになります。

```
torch.Size([37])
torch.Size([37, 6])
```

## DataLoader を作成する

グリフのアウトライン系列は可変長のため、バッチ化には `collate_fn` が必要です。`pad_sequence` を使ってバッチ内のシーケンスを揃えます。次のコードを実行してください。

```python
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(batch: list[GlyphData[Outline]]):
    types = pad_sequence([item.data.types for item in batch], batch_first=True)
    coords = pad_sequence([item.data.coords for item in batch], batch_first=True)
    return types, coords


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

loader = DataLoader(dataset, batch_size=64, shuffle=True, collate_fn=collate_fn)
types_t, coords_t = next(iter(loader))

print(types_t.shape)
print(coords_t.shape)
```

`collate_fn` はバッチ内の最長シーケンスに合わせてパディングします。実行すると次のような出力が得られます。1 次元目はバッチサイズです。2 次元目はバッチ内の最長シーケンス長で、バッチごとに異なります。

```
torch.Size([64, 369])
torch.Size([64, 369, 6])
```

## マルチプロセス読み込み

`num_workers` と `prefetch_factor` を指定すると、データ読み込みをワーカープロセスで並列化できます。シーケンス長が長いと転送コストが大きくなるため、この例の `collate_fn` で先頭 512 要素に切り詰めます。`tqdm` で全バッチを読み込んでスループットを確認します。次のコードを実行してください。

```python
from tqdm import tqdm
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.structures import GlyphData, Outline
from torchfont.transforms import LoadGlyph


def collate_fn(batch: list[GlyphData[Outline]]):
    types = pad_sequence([item.data.types[:512] for item in batch], batch_first=True)
    coords = pad_sequence([item.data.coords[:512] for item in batch], batch_first=True)
    return types, coords


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
