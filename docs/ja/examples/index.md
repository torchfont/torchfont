# サンプル集

<!-- markdownlint-disable MD013 -->

実行可能なサンプルはリポジトリの `examples/` にあります。ここでは目的別に最短で参照できるよう整理します。

::: tip 実行前に
リポジトリのルートで実行してください（例: `uv run python examples/local_fonts.py`）。

外部フォントリポジトリを使うサンプルは Git サブモジュールを使います。
`mise run data-sync` で初期化してください。

一部スクリプトは `num_workers=8` を前提にしています。`num_workers=0` にする場合は `prefetch_factor` も削除してください。
:::

## 用途別スクリプト

|用途|スクリプト (`examples/`)|要点|
|---|---|---|
|パイプライン|`local_fonts.py`|`GlyphDataset` + ローカルな `collate_fn` のオフライン例|
|バリアブルグリフ|`variable_glyphs.py`|アクセスごとにバリエーション軸の位置を一様ランダムサンプリング|
|画像パイプライン|`google_fonts.py`|Google Fonts + TorchFont と torchvision v2 の Transform + `DataLoader`|
|コーパス|`font_awesome.py`|Font Awesome のチェックアウト|
|コーパス|`material_design_icons.py`|Material Design Icons のチェックアウト|
|コーパス|`source_han_code_jp.py`|Source Han Code JP TTC のチェックアウト|

## 読む順番のおすすめ

1. `local_fonts.py`
2. `google_fonts.py`
3. `variable_glyphs.py`
4. 必要なリポジトリ向けの `font_awesome.py` / `material_design_icons.py` /
   `source_han_code_jp.py`
