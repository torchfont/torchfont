# マルチプロセス読み込み

`num_workers` と `prefetch_factor` を指定すると、データ読み込みをワーカープロセスで並列化できます。

各バッチはその中の最長 `Outline` に合わせてパディングされるため、巨大なグリフが 1 つ
混ざるとバッチ全体、ひいては学習プロセスへの転送量が膨らみます。次の例ではローカルな
`collate_fn` で各 `Outline` を 512 要素に打ち切ります。ワーカープロセスは
`collate_fn` を pickle 化するため、lambda 式ではなくモジュールレベルの関数として
定義してください。

`tqdm` で全バッチを読み込み、スループットを確認します。

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

プログレスバーの `it/s` はバッチの処理速度です。ストレージや学習環境に適した
ワーカー数とプリフェッチ設定を決める指標として利用してください。

::: tip 打ち切らずにパディングを抑える
打ち切りは幾何情報を捨てます。`Outline` 全体を保ったままパディングのコストを避けるには、
長さを打ち切るのではなく、長さを考慮した `Sampler` で近い長さのグリフをまとめてください。
:::
