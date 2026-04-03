# AGENTS.md

Prefer the repository's standard command entrypoints.

## Agent Scope

- This repository is a font-focused ML library (Python + Rust) targeting PyTorch workflows.
- Prefer stateless and thin designs across the codebase.
- Avoid over-engineered fallback paths and excessive validation.
- TorchFont is beta: prioritize better architecture over backward compatibility.

## Workflow

- If available, prefer the repository Dev Container for setup and verification.
- Do not miss hidden config directories such as `.devcontainer/`.
- If running outside a container and Dev Container CLI is available, prefer running development inside the project container.
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

## Architecture Direction

- Use fast Rust font crates (`skrifa`, `read-fonts`, etc.) for computation instead of Python-side caching.
- Do not keep runtime state in Rust objects when avoidable.
- Keep minimum required state on the Python side in pickle-friendly structures.
- Keep Rust surfaces small and deterministic, and keep Python as a thin orchestration layer.

## GitHub

- Use GitHub CLI (`gh`) for issue/PR operations.
- When possible, request Copilot review on active PRs.
- Resolve PR conversations before merging.
- Use clear branch names; avoid ambiguous prefixes like `maint`.

## Notes

- The minimum supported Python version is 3.10.
