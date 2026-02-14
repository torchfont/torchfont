---
layout: home

hero:
  name: TorchFont
  text: Learn from vector fonts<br>without rasterization
  tagline:
    "Convert TTF / OTF / TTC / OTC glyph outlines into PyTorch tensors and plug
    them into your DataLoader pipeline end-to-end."
  image:
    src: /brand/torchfont-logomark.svg
    alt: TorchFont logomark
  actions:
    - theme: brand
      text: Quickstart
      link: /en/guide/getting-started
    - theme: alt
      text: What is TorchFont?
      link: /en/guide/what-is-torchfont
    - theme: alt
      text: GitHub
      link: https://github.com/torchfont/torchfont

features:
  - icon: ⚙️
    title: Unified dataset construction
    details:
      "Use one API across local fonts (FontFolder), arbitrary Git repositories
      (FontRepo), and Google Fonts (GoogleFonts)."
  - icon: 🚀
    title: Rust backend
    details:
      "A skrifa + PyO3 backend converts glyph outlines into command + coordinate
      tensors efficiently."
  - icon: 🧱
    title: Training-ready tensor format
    details:
      "Each sample is `(types, coords, style_idx, content_idx)`, and `targets`
      provides style/content labels in one matrix."
  - icon: 🧩
    title: Composable preprocessing
    details:
      "Combine `Compose`, `LimitSequenceLength`, and `Patchify` to match your
      model input format."
---
