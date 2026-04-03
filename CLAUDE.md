# CLAUDE.md

This file gives repository-specific guidance for coding agents.

## Project and Architecture

- TorchFont is a beta Python + Rust library for font-centric ML workflows.
- Favor stateless, thin implementations.
- Prefer computing in Rust with fast font crates (`skrifa`, `read-fonts`) over Python-side caches.
- Avoid introducing mutable Rust runtime state unless there is a clear, measured need.
- Keep unavoidable Python state pickle-friendly.
- Compatibility is not a goal during this beta refactor phase.
- Avoid over-engineered fallbacks and excessive validation.

## Standard Workflow

- Prefer Dev Container-based development when available.
- Do not miss hidden configuration directories (for example `.devcontainer/`).
- Prefer `mise` tasks over ad-hoc commands.
- Run:
  - `mise run setup`
  - `mise run format`
  - `mise run check`
  - `mise run test`
- Package managers:
  - Python: `uv`
  - Rust: `cargo`
  - Docs: `npm`

## GitHub Workflow

- Use `gh` CLI for issue/PR operations.
- Request Copilot review for active PR updates when possible.
- Resolve review conversations before merge.
- Use clear branch names and avoid ambiguous prefixes such as `maint`.
