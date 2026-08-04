# Test fonts

This directory contains the smallest practical fixture set used to exercise
TorchFont's supported sfnt outline and container formats. The files are checked
in so the default test suite is deterministic and does not require network
access.

| File | Container | Outlines | Variation | Faces | Source |
| --- | --- | --- | --- | ---: | --- |
| `source-sans/SourceSans3-Regular.ttf` | Single TTF | TrueType `glyf` | Static | 1 | Adobe Source Sans |
| `source-sans/SourceSans3-Regular.otf` | Single OTF | PostScript CFF | Static | 1 | Adobe Source Sans |
| `source-serif/SourceSerif4Variable-Roman.ttf` | Single TTF | TrueType `glyf`/`gvar` | `wght`, `opsz` | 1 | Adobe Source Serif |
| `source-serif/SourceSerif4Variable-Roman.otf` | Single OTF | PostScript CFF2 | `wght`, `opsz` | 1 | Adobe Source Serif |
| `static-collection/Metropolis.ttc` | TTC | TrueType `glyf` | Static | 19 | Metropolis |
| `variable-collection/SourceHanSansVFProto.ttc` | OpenType Collection | PostScript CFF2 | `wght`, `wdth` | 6 | Adobe Variable Font Collection Test |
| `synthetic/NoOutlines-Regular.ttf` | Single TTF | None | Static | 1 | TorchFont synthetic regression fixture |

## Provenance

The upstream revisions and SHA-256 digests below make fixture updates
reproducible. License texts are stored beside the corresponding files.

### Adobe Source Sans

- Repository: <https://github.com/adobe-fonts/source-sans>
- Revision: `87b37a2daaed80fcb8e8ccb0085c4d72ddade12e`
- License: SIL Open Font License 1.1 (`source-sans/LICENSE.md`)
- `SourceSans3-Regular.ttf`:
  `4644c81b86ec9caaa76b634889968ed3c4f4f52f054855933acc7c2b21e53b0f`
- `SourceSans3-Regular.otf`:
  `08df266400933d3178d081a45f94a08814c3e55b4b7dd2e0ff69cb1329f13ab6`

### Adobe Source Serif

- Repository: <https://github.com/adobe-fonts/source-serif>
- Revision: `5f220b17d27ed64873f22cde0dd593685387bd19`
- License: SIL Open Font License 1.1 (`source-serif/LICENSE.md`)
- `SourceSerif4Variable-Roman.ttf`:
  `14d360ee1b76655da9276628b229e11671bc1f5d1083636144db6677d452cf55`
- `SourceSerif4Variable-Roman.otf`:
  `867b73c6a954a4a64616906d179f94572a748790a1d022ebeeff07f56ea0221a`

### Metropolis

- Repository: <https://github.com/typehaus/metropolis>
- Revision: `28cdaaaad51bb3d4623e17f18413a1584659fb2f`
- License: The Unlicense (`static-collection/LICENSE.md`)
- SHA-256:
  `0954cac2347bcdf4fb4d63a7fe4a460a02f31ad7fdb62764fa5995138043f0d7`

### Variable OpenType Collection

- Repository: <https://github.com/adobe-fonts/variable-font-collection-test>
- Revision: `97a0a3b3f0941ffecdc6c81d9fea59b4d76e8a5a`
- License: SIL Open Font License 1.1 (`variable-collection/LICENSE.txt`)
- SHA-256:
  `6d36aae515754374820eade8514216632b8bd324689b2306a06e279e0bc13e54`

The synthetic outline-less font is maintained by TorchFont and exists only to
verify that cmap entries without scalable outlines are filtered during dataset
indexing.
