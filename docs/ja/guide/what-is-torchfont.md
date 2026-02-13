# TorchFont とは

<!-- markdownlint-disable MD013 -->

TorchFont は、**フォントのグリフアウトラインを機械学習向けテンソルとして扱う PyTorch ライブラリ**です。画像へラスタライズしてから学習するのではなく、move / line / quadratic / cubic などのパス情報を直接扱います。

::: info
TorchFont は PyTorch の非公式ライブラリです。PyTorch プロジェクトとの公式な関係はありません。
:::

## 1分でわかる特徴

- **Dataset API を統一**:
  `FontFolder`（ローカル）/ `FontRepo`（任意 Git）/ `GoogleFonts`。
- **学習向けの出力形式**:
  `dataset[i] -> (types, coords, style_idx, content_idx)`。
- **前処理を高速化**:
  Rust バックエンド（`skrifa` + PyO3）で Python 側の変換コストを削減。
- **DataLoader 連携**:
  pickle 復元時に、ワーカー側でネイティブバックエンド状態を再構築可能。

## TorchFont が解決する課題

フォント研究では、次のようなコストがよく発生します。

- フォント収集方法が毎回バラバラで再現しにくい
- 画像化パイプラインが実験ごとに分岐して比較しにくい
- 可変フォントと静的フォントの扱いが統一されていない

TorchFont はこの部分を「収集」「テンソル化」「ラベル付与」で標準化し、モデル設計に集中しやすくします。

## 仕組み

- **Dataset 層**
  - `FontFolder`: ローカルディレクトリを走査してフォントを読み込む
  - `FontRepo`: Git リポジトリを同期してから `FontFolder` と同様に扱う
  - `GoogleFonts`: `FontRepo` の Google Fonts 向けプリセット
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

## 最小サンプル

```python
from torchfont.datasets import FontFolder

# root は実在ディレクトリである必要があります
# 例: root="~/fonts"（このリポジトリを clone 済みなら "tests/fonts" も可）
dataset = FontFolder(root="~/fonts")

types, coords, style_idx, content_idx = dataset[0]
print(types.shape)         # (seq_len,)
print(coords.shape)        # (seq_len, 6)
print(style_idx, content_idx)
```

## どんなときに向いているか

- 文字内容（content）と書体スタイル（style）を分離して学習したい
- 可変フォントを含む大規模フォント群で実験したい
- グリフ生成・分類・表現学習の入力をベクター表現で統一したい

## 次に読むページ

- [クイックスタート](/ja/guide/getting-started)
- [グリフデータ形式](/ja/guide/glyph-data-format)
- [DataLoader との統合](/ja/guide/dataloader)
