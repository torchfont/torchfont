# サンプル集

<!-- markdownlint-disable MD013 -->

実行可能なサンプルはリポジトリの `examples/` にあります。ここでは目的別に最短で参照できるよう整理します。

::: tip 実行前に
リポジトリのルートで実行してください（例: `python examples/google_fonts.py`）。

一部スクリプトは `num_workers=8` を前提にしています。`num_workers=0` にする場合は `prefetch_factor` も削除してください。
:::

## 用途別スクリプト

|用途|スクリプト (`examples/`)|要点|
|---|---|---|
|Pipeline|`google_fonts.py`|Google Fonts + Transform + DataLoader|
|Git source|`font_awesome.py`|`FontRepo` で Font Awesome|
|Git source|`material_design_icons.py`|Material Design Icons|
|Git source|`source_han_sans.py`|Source Han Sans（TTC 含む）|
|Subsetting|`subset_by_targets.py`|style/content を `targets` から抽出|

## 読む順番のおすすめ

1. `google_fonts.py`
2. `subset_by_targets.py`
3. 必要なリポジトリ向けの `font_awesome.py` / `material_design_icons.py` /
   `source_han_sans.py`
