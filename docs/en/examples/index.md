# Example Gallery

Runnable examples live in the repository `examples/` directory.

::: tip Before running
Run from the repository root, e.g. `python examples/local_fonts.py`.

Examples backed by external font repositories use Git submodules. Initialize
them with `mise run data-sync`.

Some scripts use `num_workers=8`. If you set `num_workers=0`, also remove
`prefetch_factor`.
:::

## Scripts by use case

|Use case|Script (`examples/`)|Summary|
|---|---|---|
|Pipeline|`local_fonts.py`|Offline local-font pipeline with `GlyphDataset` + local `collate_fn`|
|Corpus checkout|`google_fonts.py`|Google Fonts checkout + transforms + DataLoader|
|Corpus checkout|`font_awesome.py`|Font Awesome checkout|
|Corpus checkout|`material_design_icons.py`|Material Design Icons checkout|
|Corpus checkout|`source_han_code_jp.py`|Source Han Code JP TTC checkout|
## Suggested reading order

1. `local_fonts.py`
2. `google_fonts.py`
4. `font_awesome.py` / `material_design_icons.py` / `source_han_code_jp.py` as
   needed
