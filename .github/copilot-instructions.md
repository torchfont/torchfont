# Copilot Instructions for TorchFont

## Repository Context

- TorchFont is a beta, font-domain ML library using Python + Rust.
- Performance-critical font operations should run in Rust (fontations crates such as `skrifa` / `read-fonts`).
- Python should stay thin and orchestration-focused.

## Implementation Direction

- Prefer stateless, deterministic implementations.
- Avoid adding mutable runtime caches by default, especially in Rust.
- Keep Rust-side state minimal to avoid multiprocessing/pickle complexity.
- Keep Python-side state minimal and pickle-friendly.
- Do not add broad fallback branches or defensive validation unless explicitly required.
- Backward compatibility is not a priority for refactors in this beta phase.

## Tooling and Commands

- Prefer repository task entrypoints via `mise`.
- Setup: `mise run setup`
- Format: `mise run format`
- Lint/type-check: `mise run check`
- Test: `mise run test`
- Python package manager: `uv`
- Rust tooling: `cargo`
- Docs tooling: `npm`

## Development Environment

- Prefer using the repository Dev Container when available.
- Do not overlook hidden config directories like `.devcontainer/`.

## PR and Review Workflow

- Keep changes focused and avoid mixing unrelated refactors.
- Use `gh` for GitHub operations when scripting or reproducing CI context.
- Request Copilot review when opening/updating PRs.
- Resolve conversation threads before merge.
- Use clear branch names; avoid ambiguous prefixes such as `maint`.
