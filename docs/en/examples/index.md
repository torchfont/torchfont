# Example Gallery

Runnable examples live in the repository `examples/` directory.

::: tip Before running
Run from the repository root, e.g. `python examples/google_fonts.py`.

Some scripts use `num_workers=8`. If you set `num_workers=0`, also remove
`prefetch_factor`.
:::

## Scripts by use case

|Use case|Script (`examples/`)|Summary|
|---|---|---|
|Pipeline|`google_fonts.py`|Google Fonts + transforms + DataLoader|
|Git source|`font_awesome.py`|Font Awesome via `FontRepo`|
|Git source|`material_design_icons.py`|Material Design Icons|
|Git source|`source_han_sans.py`|Source Han Sans (TTC included)|
|Subsetting|`subset_by_targets.py`|Filter with style/content `targets`|

## Suggested reading order

1. `google_fonts.py`
2. `subset_by_targets.py`
3. `font_awesome.py` / `material_design_icons.py` / `source_han_sans.py` as
   needed
