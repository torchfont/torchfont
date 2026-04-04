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
    title: Local-first dataset API
    details:
      "Point `GlyphDataset(root=...)` at any local font directory or
      already-cloned repository checkout."
  - icon: 🚀
    title: Rust backend
    details:
      "A skrifa + PyO3 backend converts glyph outlines into command + coordinate
      tensors efficiently."
  - icon: 🧱
    title: Sample + batch primitives
    details:
      "`GlyphSample` represents one glyph with outline, metrics, and name.
      `collate_fn` builds a padded `GlyphBatch` ready for training."
  - icon: 🧩
    title: Composable preprocessing
    details:
      "Combine `Compose`, `LimitSequenceLength`, and `Patchify` to match your
      model input format."
---
