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
    details: "skrifa + PyO3 による実装で、グリフアウトラインを command + coordinate テンソルへ高速に変換します。"
  - icon: 🧱
    title: Sample / Batch の基本型
    details: "`GlyphSample` が 1 グリフ、`collate_fn` が padded tensor と mask を持つ `GlyphBatch` を返します。"
  - icon: 🧩
    title: 合成可能な前処理
    details: "`Compose` / `LimitSequenceLength` / `Patchify` を組み合わせ、モデルに合わせた入力形式へ調整できます。"
---
