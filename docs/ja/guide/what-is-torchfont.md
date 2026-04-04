# TorchFont とは

<!-- markdownlint-disable MD013 -->

TorchFont は、**フォントのグリフアウトラインを機械学習向けテンソルとして扱う PyTorch ライブラリ**です。画像へラスタライズしてから学習するのではなく、move / line / quadratic / cubic などのパス情報を直接扱います。

::: info
TorchFont は PyTorch の非公式ライブラリです。PyTorch プロジェクトとの公式な関係はありません。
:::

## 1分でわかる特徴

- **中心となる Dataset API**:
  `GlyphDataset(root=...)` が、ローカルのフォントディレクトリや
  checkout 済みリポジトリをそのまま読み込みます。
- **sample-first な出力形式**:
  `dataset[i] -> GlyphSample(types, coords, style_idx, content_idx, metrics, glyph_name)`。
- **組み込みのバッチ化**:
  `torchfont.utils.collate_fn(batch) -> GlyphBatch`。
- **前処理を高速化**:
  Rust バックエンド（`skrifa` + PyO3）で Python 側の変換コストを削減。
- **DataLoader 連携**:
  pickle 復元時に、ワーカー側でネイティブバックエンド状態を再構築可能。

## TorchFont が解決する課題

フォント研究では、次のようなコストがよく発生します。

- フォント収集と glyph 読み出しの境界が曖昧で、実験系が散らばりやすい
- 画像化パイプラインが実験ごとに分岐して比較しにくい
- 可変フォントと静的フォントの扱いが統一されていない

TorchFont はこの部分を「テンソル化」「ラベル付与」「バッチ化」で標準化し、
モデル設計に集中しやすくします。

## 仕組み

- **Dataset 層**
  - `GlyphDataset`: ローカルディレクトリを走査してフォントを読み込む
  - Git リポジトリも、checkout 済みなら通常のディレクトリとして扱える
- **Rust バックエンド**
  - フォントの charmap から codepoint と glyph を対応付け
  - アウトラインをコマンド列と 6 次元座標列へ変換
  - 座標は `units_per_em` で正規化
  - 2 次ベジェと 3 次ベジェを別コマンドとして保持
- **Transform 層**
  - `QuadToCubic`: `QUAD_TO` を `CURVE_TO` に統一
  - `LimitSequenceLength`: 長いシーケンスを切り詰め
  - `Patchify`: 固定長パッチへ再構成
  - `Compose`: 前処理を順に合成
- **Batch utility**
  - `collate_fn`: 可変長サンプルを `GlyphBatch` へまとめる
  - `GlyphBatch.targets`: style・content のインデックスを `(B, 2)` テンソルで保持
  - `GlyphBatch.metrics`: サンプルごとのメトリクスを `(B, 15)` float テンソルで保持

## 最小サンプル

```python
from torchfont.datasets import GlyphDataset

# root は実在ディレクトリである必要があります
# 例: root="~/fonts"（このリポジトリを clone 済みなら "tests/fonts" も可）
dataset = GlyphDataset(root="~/fonts")

sample = dataset[0]
print(sample.types.shape)         # (seq_len,)
print(sample.coords.shape)        # (seq_len, 6)
print(sample.style_idx, sample.content_idx)
print(sample.glyph_name)
```

## どんなときに向いているか

- 文字内容（content）と書体スタイル（style）を分離して学習したい
- 可変フォントを含む大規模フォント群で実験したい
- グリフ生成・分類・表現学習の入力をベクター表現で統一したい

## 次に読むページ

- [クイックスタート](/ja/guide/getting-started)
- [グリフデータ形式](/ja/guide/glyph-data-format)
- [DataLoader との統合](/ja/guide/dataloader)
