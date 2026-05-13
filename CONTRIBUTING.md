# Contributing to TorchFont

Thank you for taking the time to improve TorchFont. The guidelines below keep
the project healthy and make it easier for maintainers to review changes.

## Project Setup

TorchFont uses `mise` for tool versions and repeatable project tasks. Use the
Dev Container when it is available, then install dependencies with:

```bash
mise run sync
```

This installs the Python, Rust, and Node tooling used by the repository. For
data-backed examples or tests, sync submodules with:

```bash
mise run data-sync
```

## Coding Standards

- The minimum supported Python version is 3.10. Avoid syntax that would break on
  that interpreter.
- Keep Python thin, typed, and pickle-friendly; put font parsing and heavier
  deterministic computation in Rust.
- Public APIs live under `torchfont.datasets`, `torchfont.transforms`, and
  related metadata helpers.
- Avoid broad fallback paths or hidden network/git behavior unless they protect
  a real external boundary.

## Formatting, Checks, and Tests

Run the project tasks before requesting review:

```bash
mise run format
mise run check
mise run test
```

`mise run check` covers Rust formatting, clippy, cargo check, Ruff, and `ty`.
`mise run test` builds the Rust extension with `maturin develop` before running
pytest.

By default, pytest skips slow and network-dependent tests. To include them, use:

```bash
uv run pytest tests --runslow
uv run pytest tests --runnetwork
uv run pytest tests --runslow --runnetwork
```

## Documentation

Docs use VitePress. Build them with:

```bash
mise run docs-build
```

Keep `docs/en/` and `docs/ja/` aligned when changing user-facing docs.

## Git Workflow

- Create topic branches off `main`.
- Write descriptive commit messages. Mention the relevant issue when applicable.
- Keep pull requests focused. Separate unrelated refactors or formatting changes
  into their own PRs.
- Ensure CI passes before requesting review.

## Need Help?

Open a GitHub Discussion or issue if anything here is unclear. The more context
you provide, such as logs, screenshots, or sample fonts, the faster reviewers can
help.
