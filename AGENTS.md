# AGENTS.md

These guidelines apply to everyday development. See
[CONTRIBUTING.md](CONTRIBUTING.md) for details and task-specific instructions.
If the two documents differ, follow `CONTRIBUTING.md`.

## Tools

- Use mise to run development tasks and manage development tools.
- Use uv for Python dependencies and commands, Cargo for Rust, and maturin for Python/Rust integration.
- Manage PyTorch using uv's PyTorch integration.
- Use the GitHub CLI for GitHub operations.

## Design

- Avoid overengineering and prefer existing libraries over custom implementations.
- Backward compatibility is not required during beta. Do not add compatibility aliases or fallbacks.
- Follow "Parse, don't validate": validate only at external boundaries and avoid excessive exception handling.
- Follow PyTorch ecosystem conventions and support features such as `torch.compile`.
- Prefer standard PyTorch types and protocols. Return bitmaps as ordinary tensors and do not depend on TorchVision at runtime.
- Do not modify shared PyTorch settings to change the default behavior of components such as DataLoader. Let users explicitly choose any custom collation behavior.
- Keep Rust stateless and deterministic, Python objects picklable, and stochastic operations in Python.

## Code Checks

- Run formatting, linting, type checks, and tests. Keep tests focused and avoid redundant coverage.

## Documentation

- Do not add source comments or private API docstrings. Keep documentation and public docstrings focused on library users.
- Keep Japanese and English documentation aligned and consistent with existing documentation.
