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
    title: ローカルファーストなデータセット API
    details: "`GlyphDataset(root=...)` を使い、ローカルのフォントディレクトリやクローン済みのリポジトリをそのまま入力にできます。"
  - icon: 🚀
    title: Rust バックエンド
    details: "`skrifa` と PyO3 による実装でフォントをインデックス化し、必要なときにグリフアウトラインを要素型と座標のテンソルへ読み込みます。"
  - icon: 🧱
    title: 参照を中心としたデータモデル
    details: "`GlyphSample` は決定的なグリフ参照とターゲットインデックスを持ち、読み込みやバッチ化の方針は `transform` 側に置けます。"
  - icon: 🧩
    title: 柔軟な前処理
    details: "`torchvision.transforms.v2` 形式の意味型変換をパイプラインとして組み合わせ、可変長テンソルの整形はモデル固有の `collate_fn` で調整できます。"
---
