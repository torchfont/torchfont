# Google Fonts を使う

<!-- markdownlint-disable MD013 -->

`GoogleFonts` は、Google Fonts リポジトリを shallow fetch して Dataset 化するためのプリセットです。

## 最小例

```python
from torchfont.datasets import GoogleFonts

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    download=True,
    depth=1,
)

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")
print(f"commit={dataset.commit_hash}")
```

## 実務フロー

### 1. 初回同期（ネットワークあり）

```python
_ = GoogleFonts(root="data/google/fonts", ref="main", download=True)
```

### 2. ローカルキャッシュを再利用

```python
reused = GoogleFonts(root="data/google/fonts", ref="main", download=False)
print(reused.commit_hash)
```

## `download` の挙動

| モード             | ネットワーク fetch | `ref` の解決元            |
| ------------------ | ------------------ | ------------------------- |
| `download=True`    | あり               | リモート fetch の結果     |
| `download=False`   | なし               | ローカル Git オブジェクト |

`download` が制御するのは fetch の有無だけです。どちらのモードでも TorchFont は `root` に対して `ref` を force checkout します。

`root/.git` がない初回に `download=False` を指定すると `FileNotFoundError` になります。新しいキャッシュディレクトリごとに、最初の 1 回は `download=True` を使ってください。

`depth` は fetch 履歴の深さを制御します（既定は `1` の shallow、`0` で履歴全体）。`download=True` の `ref` は具体的なブランチ参照（`main` または `refs/heads/main`）か明示 `refs/...` を使ってください。remote-tracking ref（`origin/main`）と revspec（リビジョン指定, 例: `main~1`）は受け付けません。

## Google Fonts 専用の `root` を使う

`GoogleFonts` は、Google Fonts の URL を固定した `FontRepo` プリセットです。

`root/.git` がすでにある場合は、その既存リポジトリを再利用し、既存 `origin` URL がこの固定 URL と一致している必要があります。`data/google/fonts` のような専用キャッシュディレクトリを使い、別ソースと共有しないでください。既存リポジトリに `origin` remote がない場合は `ValueError` で失敗します。

::: warning
`root` はキャッシュ専用ディレクトリとして使ってください。`root` 配下のローカル変更は初期化時に上書きされる可能性があります。
:::

## 取得対象を絞る

### 既定の `patterns`

`patterns=None` のときは次が使われます。

```python
(
    "apache/*/*.ttf",
    "ofl/*/*.ttf",
    "ufl/*/*.ttf",
    "!ofl/adobeblank/AdobeBlank-Regular.ttf",
)
```

リポジトリ全体ではなく一部だけを使う場合は、`patterns` を明示してください。

### 文字集合を絞る

```python
dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    codepoint_filter=range(0x30, 0x3A),  # 0-9
    download=True,
    depth=1,
)
```

`codepoint_filter` を使うと、不要な文字サンプルを初期化時に除外できます。

## 再現性のための運用

`main` のようなブランチ名は時間とともに移動します。厳密に再現したい場合は `ref` にコミットハッシュを使うか、`dataset.commit_hash` を保存して再利用します。

```python
dataset = GoogleFonts(root="data/google/fonts", ref="main", download=True)
print(dataset.commit_hash)

# 後で同じスナップショットを再現
repro = GoogleFonts(
    root="data/google/fonts",
    ref=dataset.commit_hash,
    download=False,
)
```

## 学習用パイプライン例

```python
from collections.abc import Sequence
import sys

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import GoogleFonts
from torchfont.transforms import Compose, LimitSequenceLength, Patchify

transform = Compose([
    LimitSequenceLength(max_len=512),
    Patchify(patch_size=32),
])

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    transform=transform,
    download=True,
)


def collate_fn(
    batch: Sequence[tuple[Tensor, Tensor, int, int]],
) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    types_list = [t for t, _, _, _ in batch]
    coords_list = [c for _, c, _, _ in batch]
    style_list = [s for _, _, s, _ in batch]
    content_list = [c for _, _, _, c in batch]

    types = pad_sequence(types_list, batch_first=True, padding_value=0)
    coords = pad_sequence(coords_list, batch_first=True, padding_value=0.0)

    style_idx = torch.as_tensor(style_list, dtype=torch.long)
    content_idx = torch.as_tensor(content_list, dtype=torch.long)
    return types, coords, style_idx, content_idx


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

- Linux: `"fork"` が高速になりやすい
- macOS: `"spawn"` または `"forkserver"`
- Windows: `"spawn"`
- `num_workers=0` にする場合は `prefetch_factor` と `multiprocessing_context` を指定しないでください

:::

## 関連ページ

- [Git リポジトリからのフォント取得](/ja/guide/git-repos)
- [DataLoader との統合](/ja/guide/dataloader)
- [サブセット抽出](/ja/guide/subset)
