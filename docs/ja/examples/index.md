# サンプル集

<!-- markdownlint-disable MD013 -->

実行可能なサンプルはリポジトリの `examples/` にあります。ここでは目的別に最短で参照できるよう整理します。

::: tip 実行前に
リポジトリのルートで実行してください（例: `uv run python examples/local_fonts.py`）。

外部フォントリポジトリを使うサンプルは Git submodule を使います。
`mise run data-sync` で初期化してください。

一部スクリプトは `num_workers=8` を前提にしています。`num_workers=0` にする場合は `prefetch_factor` も削除してください。
:::

## 用途別スクリプト

|用途|スクリプト (`examples/`)|要点|
|---|---|---|
|Pipeline|`local_fonts.py`|`GlyphDataset` + ローカルな `collate_fn` のオフライン例|
|Variable glyph|`variable_glyphs.py`|アクセスごとに variation axis の location を一様ランダムサンプリング|
|画像pipeline|`google_fonts.py`|Google Fonts checkout + TorchFontとtorchvision v2のtransform + DataLoader|
|Corpus checkout|`font_awesome.py`|Font Awesome の checkout|
|Corpus checkout|`material_design_icons.py`|Material Design Icons の checkout|
|Corpus checkout|`source_han_code_jp.py`|Source Han Code JP TTC の checkout|

## 読む順番のおすすめ

1. `local_fonts.py`
2. `google_fonts.py`
3. `variable_glyphs.py`
4. 必要なリポジトリ向けの `font_awesome.py` / `material_design_icons.py` /
   `source_han_code_jp.py`
