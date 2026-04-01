# AGENTS.md

Prefer the repository's standard command entrypoints.

## Workflow

- If available, prefer the repository Dev Container for setup and verification.
- Run `mise run setup` for local setup.
- Prefer `mise` tasks when an equivalent task exists.
- The repository uses `uv` for Python, `cargo` for Rust, and `npm` for documentation tooling.
- Add new repeatable project-wide workflows to `mise.toml`.
- Check nested `AGENTS.md` files for path-specific instructions.

## Standard Commands

- Format: `mise run format`
- Lint and type-check: `mise run check`
- Test: `mise run test`

## Expectations

- TorchFont is beta; do not preserve compatibility or add fallbacks.
- After code edits, run `mise run format`.
- Before finishing Python or Rust changes, run `mise run check`.
- If behavior changes, run relevant tests in addition to the standard checks.
- Keep changes focused; avoid mixing unrelated refactors with behavior changes.

## Notes

- The minimum supported Python version is 3.10.
