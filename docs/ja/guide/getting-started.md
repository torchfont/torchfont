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

`GlyphDataset` はローカルディレクトリ配下のフォント（`.ttf` / `.otf` /
`.ttc` / `.otc`）を走査して Dataset を作ります。

```python
from torchfont.datasets import GlyphDataset

# root は存在するディレクトリを指定してください
# 例: root="~/fonts"（このリポジトリを clone 済みなら "tests/fonts" も可）
dataset = GlyphDataset(root="~/fonts")

print(f"samples={len(dataset)}")
print(f"styles={len(dataset.style_classes)}")
print(f"contents={len(dataset.content_classes)}")

sample = dataset[0]
print(sample.types.shape)   # (seq_len,)
print(sample.coords.shape)  # (seq_len, 6)
```

::: warning
`GlyphDataset` は `root` を絶対パス化してから読み込みます。存在しないパスを渡すと初期化時に例外になります。
:::

## 3. DataLoader に渡す

グリフは可変長なので、組み込みの `torchfont.utils.collate_fn` を使うのが基本です。

```python
from torch.utils.data import DataLoader

from torchfont.datasets import GlyphDataset
from torchfont.utils import collate_fn


dataset = GlyphDataset(root="~/fonts")
loader = DataLoader(dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)

batch = next(iter(loader))
print(batch.types.shape)
print(batch.coords.shape)
print(batch.mask.shape)
```

## 4. ローカル checkout を大きなコーパスに向ける

リポジトリの同期は TorchFont の外側で行い、その checkout を
`GlyphDataset` に渡します。

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

print(len(dataset), len(dataset.style_classes), len(dataset.content_classes))
```

TorchFont は checkout 済みディレクトリを通常のローカルフォルダとして扱います。
Git などで更新したあとは Dataset インスタンスを作り直してください。

## よくある最初の改善

- 長すぎるシーケンスを制限する: `LimitSequenceLength(max_len=...)`
- 2 次セグメントを 3 次へ統一する: `QuadToCubic()`
- 固定長の入力へ揃える: `Patchify(patch_size=...)`
- 学習対象を絞る: `codepoint_filter=` や `patterns=`

## 次に読むページ

- [グリフデータ形式](/ja/guide/glyph-data-format)
- [DataLoader との統合](/ja/guide/dataloader)
- [Git checkout 済みリポジトリを使う](/ja/guide/git-repos)
- [Google Fonts](/ja/guide/google-fonts)
