"""TorchFont: A PyTorch-native toolkit for modeling and processing vector fonts.

Notes:
    TorchFont is local-first. You point it at a directory of font files or an
    already-synced repository checkout on disk, and it returns glyph samples
    and collated glyph batches suitable for PyTorch training code.

Features:
    * A primary ``GlyphDataset`` API for local font directories and checkouts.
    * A Rust backend that renders glyph outlines directly into PyTorch-ready
      tensors.
    * Small transform utilities for adapting glyph samples.
    * A built-in ``collate_outline`` that pads ``(types, coords)`` pairs into
      batch tensors.

Examples:
    Assemble a dataset from local fonts::

        from torchfont.datasets import GlyphDataset

        ds = GlyphDataset(root="~/fonts")

    Build padded batches with the utility module::

        from torchfont.utils import collate_outline

References:
    The project README covers installation, advanced usage, and contribution
    guidelines in greater depth.

Package Layout:
    Core public APIs live in submodules such as ``torchfont.datasets``,
    ``torchfont.utils``, ``torchfont.transforms``, and ``torchfont.io``.

"""

__all__: list[str] = []
