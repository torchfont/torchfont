---
layout: home

hero:
  name: TorchFont
  text: ベクターフォントを<br>そのまま学習データへ
  tagline: "TTF / OTF / TTC / OTC のグリフアウトラインを PyTorch テンソルへ変換し、DataLoader まで一貫して扱えるライブラリ"
  image:
    src: /brand/torchfont-logomark.svg
    alt: TorchFont logomark
  actions:
    - theme: brand
      text: クイックスタート
      link: /ja/guide/getting-started
    - theme: alt
      text: TorchFont とは
      link: /ja/guide/what-is-torchfont
    - theme: alt
      text: GitHub
      link: https://github.com/torchfont/torchfont

features:
  - icon: ⚙️
    title: データセット構築を統一
    details: "ローカルフォント（FontFolder）、任意 Git リポジトリ（FontRepo）、Google Fonts（GoogleFonts）を同じ API で扱えます。"
  - icon: 🚀
    title: Rust バックエンド
    details: "skrifa + PyO3 による実装で、グリフアウトラインを command + coordinate テンソルへ高速に変換します。"
  - icon: 🧱
    title: 学習向けテンソル形式
    details: "各サンプルは `(types, coords, style_idx, content_idx)`。`targets` によりスタイル/文字ラベルをまとめて利用できます。"
  - icon: 🧩
    title: 合成可能な前処理
    details: "`Compose` / `LimitSequenceLength` / `Patchify` を組み合わせ、モデルに合わせた入力形式へ調整できます。"
---
