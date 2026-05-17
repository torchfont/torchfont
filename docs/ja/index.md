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
    title: ローカルファーストな Dataset API
    details: "`GlyphDataset(root=...)` を使い、ローカルのフォントディレクトリや clone 済み checkout をそのまま入力にできます。"
  - icon: 🚀
    title: Rust バックエンド
    details: "skrifa + PyO3 による実装で、グリフアウトラインを element type + coordinates テンソルへ高速に変換します。"
  - icon: 🧱
    title: Sample-first なデータモデル
    details: "`GlyphSample` が 1 グリフを outline・metrics・名前つきで表し、batch 化方針は学習コード側に置けます。"
  - icon: 🧩
    title: 柔軟な前処理
    details: "`quad_to_cubic` のような小さな utility を使い、tensor 整形は dataset やモデル固有コードで調整できます。"
---
