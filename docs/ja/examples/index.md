# サンプル集

<!-- markdownlint-disable MD013 -->

実行可能なサンプルはリポジトリの `examples/` にあります。ここでは目的別に最短で参照できるよう整理します。

::: tip 実行前に
リポジトリのルートで実行してください（例: `python examples/local_fonts.py`）。

一部スクリプトは `num_workers=8` を前提にしています。`num_workers=0` にする場合は `prefetch_factor` も削除してください。
:::

## 用途別スクリプト

|用途|スクリプト (`examples/`)|要点|
|---|---|---|
|Pipeline|`local_fonts.py`|`GlyphDataset` + 組み込み `collate_fn` のオフライン例|
|Corpus checkout|`google_fonts.py`|Google Fonts checkout + Transform + DataLoader|
|Corpus checkout|`font_awesome.py`|Font Awesome の checkout|
|Corpus checkout|`material_design_icons.py`|Material Design Icons の checkout|
|Corpus checkout|`source_han_sans.py`|Source Han Sans の checkout（TTC 含む）|
|Subsetting|`subset_by_targets.py`|style/content を `targets` から抽出|

## 読む順番のおすすめ

1. `local_fonts.py`
2. `subset_by_targets.py`
3. `google_fonts.py`
4. 必要なリポジトリ向けの `font_awesome.py` / `material_design_icons.py` /
   `source_han_sans.py`
