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

## Transform Architecture

Transform modules are organized by font-domain responsibility:

- `torchfont` exports core semantic values such as `Outline` and `GlyphData`.
  Rasterized glyphs remain plain tensors and enter image semantics explicitly
  through TorchVision.
- `torchfont.transforms._transform` contains only the transform engine, while
  `_container` contains composition primitives.
- Class transforms are split into `_glyph`, `_curves`, `_geometry`, `_outline`,
  `_subpath`, and `_bitmap` modules.
- `torchfont.transforms.functional` mirrors those domains. Its public functions
  are deterministic semantic kernels; `_utils` contains shared helpers for the
  Rust/NumPy boundary.

Keep class transforms configuration-only and use `make_params()` for random
sampling. Put deterministic behavior in the corresponding functional module so
class and direct functional calls share one kernel implementation. Add a new
domain module only when the operation does not fit an existing font concept.
Do not add a kernel registry until more than one semantic representation needs
dispatch.

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

## Documentation

Docs use VitePress. Build them with:

```bash
npm ci
npm run docs:build
```

Keep `docs/en/` and `docs/ja/` aligned when changing user-facing docs.

## Git Workflow

- Create topic branches off `main`.
- Write descriptive commit messages. Mention the relevant issue when applicable.
- Keep pull requests focused. Separate unrelated refactors or formatting changes
  into their own PRs.
- Ensure CI passes before requesting review.

## Release

### Binary Build Matrix

TorchFont publishes wheels for the intersection of the targets supported by
maturin and PyO3, prebuilt Skia, and PyTorch.

| Platform | Rust target | maturin / PyO3 | Prebuilt Skia | PyTorch | **TorchFont** |
| --- | --- | :---: | :---: | :---: | :---: |
| manylinux x86-64 | `x86_64-unknown-linux-gnu` | ✅ | ✅ | ✅ | **✅** |
| manylinux x86 | `i686-unknown-linux-gnu` | ✅ | ❌ | ❌ | **❌** |
| manylinux ARM64 | `aarch64-unknown-linux-gnu` | ✅ | ✅ | ✅ | **✅** |
| manylinux ARMv7 | `armv7-unknown-linux-gnueabihf` | ✅ | ❌ | ❌ | **❌** |
| manylinux s390x | `s390x-unknown-linux-gnu` | ✅ | ❌ | ❌ | **❌** |
| manylinux PowerPC 64 LE | `powerpc64le-unknown-linux-gnu` | ✅ | ❌ | ❌ | **❌** |
| musllinux x86-64 | `x86_64-unknown-linux-musl` | ✅ | ❌ | ❌ | **❌** |
| musllinux x86 | `i686-unknown-linux-musl` | ✅ | ❌ | ❌ | **❌** |
| musllinux ARM64 | `aarch64-unknown-linux-musl` | ✅ | ❌ | ❌ | **❌** |
| musllinux ARMv7 | `armv7-unknown-linux-musleabihf` | ✅ | ❌ | ❌ | **❌** |
| Windows x86-64 | `x86_64-pc-windows-msvc` | ✅ | ✅ | ✅ | **✅** |
| Windows x86 | `i686-pc-windows-msvc` | ✅ | ❌ | ❌ | **❌** |
| Windows ARM64 | `aarch64-pc-windows-msvc` | ✅ | ✅ | ❌ | **❌** |
| macOS Intel | `x86_64-apple-darwin` | ✅ | ✅ | ❌ | **❌** |
| macOS Apple silicon | `aarch64-apple-darwin` | ✅ | ✅ | ✅ | **✅** |
| Emscripten | `wasm32-unknown-emscripten` | ✅ | ✅ | ❌ | **❌** |
| Android ARM64 | `aarch64-linux-android` | ✅ | ✅ | ❌ | **❌** |
| Android x86-64 | `x86_64-linux-android` | ✅ | ✅ | ❌ | **❌** |

Use these upstream references when reviewing the matrix:

- [maturin distribution guide](https://www.maturin.rs/distribution.html) for
  packaging targets
- [rust-skia platform support](https://github.com/rust-skia/rust-skia#platform-support-build-targets-and-prebuilt-binaries)
  for prebuilt Skia binaries
- [PyTorch binary support policy](https://github.com/pytorch/pytorch/blob/main/RELEASE.md)
  for officially supported platforms

### Release Process

1. Open a release pull request that updates the version in `Cargo.toml` and
   `Cargo.lock`, and the version and release date in `CITATION.cff`.
2. Merge the pull request into `main` after CI passes.
3. Create a GitHub Release from `main` with a matching `vX.Y.Z` tag and
   GitHub-generated release notes.
4. Confirm that the tag-triggered CI builds, attests, and publishes the wheels
   and source distribution to PyPI.
5. Verify the published GitHub Release and the wheel and source distributions
   for the new version on PyPI.

## Need Help?

Open a GitHub Discussion or issue if anything here is unclear. The more context
you provide, such as logs, screenshots, or sample fonts, the faster reviewers can
help.
