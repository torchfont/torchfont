# AGENTS.md

## Architecture

- TorchFont is a beta Python + Rust library for font-focused PyTorch workflows; prefer better architecture over backward compatibility.
- Keep Python thin and pickle-friendly; put font parsing and computation in deterministic Rust using crates such as `skrifa` and `read-fonts`.
- Avoid mutable Rust runtime state unless the design clearly requires it.
- Avoid broad fallback paths and excessive validation unless they protect a real external boundary.

## Workflow

- Prefer the Dev Container when available.
- Prefer existing `mise` tasks over ad hoc commands.
- Use `uv` for all Python operations; never invoke `python` or `pip` directly.
- Run formatting, checks, and relevant tests after code changes.
- Docs use VitePress. Keep `docs/en/` and `docs/ja/` aligned.

## GitHub

- Use `gh` for issue and pull request operations.
- Resolve PR conversations after addressing the feedback.

## Known non-starters

- **Making `skia-safe` an optional Cargo feature**: PyPI wheels are compiled
  binaries — Cargo features cannot be selected via `pip install`. Truly optional
  Skia would require a separate distribution package, which is not planned.
