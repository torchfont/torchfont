# クイックスタート

<!-- markdownlint-disable MD013 -->

このページでは、TorchFont を導入して最初の Dataset と DataLoader を動かすまでを扱います。

## 前提

- Python 3.10 以上
- PyTorch 2.3 以上

## 1. インストール

::: code-group

```bash [uv (推奨)]
uv add torchfont
```

```bash [pip]
pip install torchfont
```

:::

## 2. ローカルフォントを読む

`FontFolder` はローカルディレクトリ配下のフォント（`.ttf` / `.otf` / `.ttc` / `.otc`）を走査して Dataset を作ります。

```python
from torchfont.datasets import FontFolder

# root は存在するディレクトリを指定してください
# 例: root="~/fonts"（このリポジトリを clone 済みなら "tests/fonts" も可）
dataset = FontFolder(root="~/fonts")

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")

types, coords, style_idx, content_idx = dataset[0]
print(types.shape)   # (seq_len,)
print(coords.shape)  # (seq_len, 6)
```

::: warning
`FontFolder` は `root` を絶対パス化してから読み込みます。存在しないパスを渡すと初期化時に例外になります。
:::

## 3. DataLoader に渡す

グリフは可変長なので、`collate_fn` でパディングするのが基本です。

```python
from collections.abc import Sequence

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader

from torchfont.datasets import FontFolder


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


dataset = FontFolder(root="~/fonts")
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
print([x.shape for x in batch[:2]])
```

## 4. Google Fonts で大規模に試す

```python
from torchfont.datasets import GoogleFonts

dataset = GoogleFonts(
    root="data/google/fonts",
    ref="main",
    download=True,
)

print(dataset.commit_hash)
print(len(dataset), len(dataset.style_classes), len(dataset.content_classes))
```

- `download=True`: リモートから fetch し、`ref` を force checkout
- `download=False`: fetch を省略し、ローカルで `ref` を解決して force checkout（ローカルで解決可能な `ref` が必要）
- `depth=1`: shallow fetch（既定）、`depth=0`: 履歴全体を取得
- `download=True` の `ref` は具体的なブランチ参照（`main` または `refs/heads/main`）か、明示 `refs/...` のみを想定します。remote-tracking ref（`origin/main`）や revspec（リビジョン指定, 例: `main~1`）は受け付けません。

`root/.git` がない初回に `download=False` を指定すると `FileNotFoundError` になります。新しいキャッシュディレクトリでは最初に一度 `download=True` で同期してください。

::: warning
`FontRepo` / `GoogleFonts` は `root` を `ref` に合わせるために force checkout を行います。`root` はデータセット用キャッシュとして使い、手作業の変更を混在させないでください。
:::

## よくある最初の改善

- 長すぎるシーケンスを制限する: `LimitSequenceLength(max_len=...)`
- 2 次セグメントを 3 次へ統一する: `QuadToCubic()`
- 固定長の入力へ揃える: `Patchify(patch_size=...)`
- 学習対象を絞る: `codepoint_filter=` や `patterns=`

## 次に読むページ

- [グリフデータ形式](/ja/guide/glyph-data-format)
- [DataLoader との統合](/ja/guide/dataloader)
- [Google Fonts](/ja/guide/google-fonts)
