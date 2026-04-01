# AGENTS.md

This file applies to `tests/`.

- `mise run test` runs the default offline test suite.
- Tests marked `slow` or `network` are skipped unless `--runslow` or `--runnetwork` is passed.
- When changing behavior covered by slow or network tests, run the relevant tests explicitly with `uv run pytest`.
- Keep test runs targeted to the affected modules when practical.
