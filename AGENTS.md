# AGENTS.md

## Architecture

- TorchFont is a beta Python + Rust library for font-focused PyTorch workflows; prefer better architecture over backward compatibility.
- Keep Python thin and pickle-friendly; put font parsing and computation in deterministic Rust using crates such as `skrifa` and `read-fonts`.
- Avoid mutable Rust runtime state unless the design clearly requires it.
- Add validation only to prevent **silent data corruption** at real external boundaries (user files and arguments). Do not validate when: invalid input would trigger a loud error in a dependency or subsequent operation; a dependency already rejects the value; or the documented behavior for invalid input is a reasonable empty/no-op result. Validate once at the entry point.

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
- **Sharing parsed font state across instance functions, indexing, and loading**:
  keep these boundaries simple and pickle-friendly. Re-parsing is acceptable;
  measured savings were only about 70–85 ms, or 0.34% for full indexing.
