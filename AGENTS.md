# AGENTS.md

## Architecture

- TorchFont is beta; ignore backward compatibility. Add no shims, deprecated aliases, version branches, or fallbacks. Raise minimum dependencies instead.
- Follow current public PyTorch and TorchVision design, including autograd, `torch.compile`, device/dtype, `nn.Module`, and `Dataset`/`DataLoader` conventions where applicable. Add no abstraction, implicit behavior, or convenience API without a clear analogue.
- Prefer standard PyTorch types and protocols. Add custom types only for otherwise inexpressible font invariants. Never mutate global PyTorch registries or prescribe collation.
- Return bitmaps as ordinary Tensor data usable by TorchVision, but never import or depend on TorchVision at runtime. Development and interoperability tests may use it.
- Follow current PyTorch and TorchVision naming, visibility, modules, and directory layout; ignore legacy structure kept for compatibility.
- Keep Python thin and pickle-friendly; put font parsing and computation in deterministic Rust using crates such as `skrifa` and `read-fonts`.
- Avoid mutable Rust runtime state unless the design clearly requires it.
- Validate once at external boundaries only to prevent **silent data corruption**. Rely on dependencies and downstream operations to raise; allow documented empty/no-op results.

## Workflow

- Prefer the Dev Container when available.
- Prefer existing `mise` tasks over ad hoc commands.
- Use `uv` for all Python operations; never invoke `python` or `pip` directly.
- Run formatting, checks, and relevant tests after code changes.
- Docs use VitePress. Document only the current public API for users—never history, internals, compatibility notes, or maintainer guidance. Keep `docs/en/` and `docs/ja/` aligned.

## GitHub

- Use `gh` for issue and pull request operations.
- Resolve PR conversations after addressing the feedback.

## Known non-starters

- **Making `skia-safe` an optional Cargo feature**: PyPI wheels are compiled
  binaries — Cargo features cannot be selected via `pip install`. Truly optional
  Skia would require a separate distribution package, which is not planned.
- **Sharing parsed font state across indexing, axis inspection, and loading**:
  keep these boundaries simple and pickle-friendly. Re-parsing is acceptable;
  measured savings were only about 70–85 ms, or 0.34% for full indexing.
