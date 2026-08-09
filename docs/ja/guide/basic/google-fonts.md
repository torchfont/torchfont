# Google Fonts のセットアップ

## Google Fonts とは

[Google Fonts](https://fonts.google.com/) は、多様なスタイルと文字体系を含む大規模な
フォントコレクションです。

- **幅広い収録範囲**: 多数のフォントファミリー、スタイル、文字体系を含みます。
- **リポジトリ構成**: このガイドで使用するフォントファイルは、`apache`、`ofl`、`ufl`
  ディレクトリに分類されています。
- **データの版管理**: 実験に使用したリビジョンを Git のコミットで特定できます。

フォントを使用または再配布する前に、それぞれのライセンスを確認してください。

## リポジトリをクローンする

このガイドのサンプルで使用するパスへ Google Fonts をクローンします。
`--depth 1` で最新のコミットのみを取得し、ダウンロードサイズを抑えます。

```bash
git clone --depth 1 https://github.com/google/fonts.git data/google/fonts
```
