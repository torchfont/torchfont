"""TorchFont: A PyTorch-native toolkit for modeling and processing vector fonts.

Notes:
    TorchFont ships cohesive building blocks—dataset wrappers, a compiled glyph
    renderer, and preprocessing transforms—that keep glyph-centric machine
    learning pipelines declarative and reproducible.

Features:
    * Seamless Google Fonts integration backed by shallow Git clones and
      pattern-aware font discovery.
    * A Rust backend that renders glyph outlines directly into PyTorch-ready
      tensors.
    * Composable transform primitives for truncation, batching, and patch-based
      reshaping.

Examples:
    Assemble a dataset sourced from Google Fonts::

        from torchfont.datasets import GoogleFonts

        ds = GoogleFonts(root="data/google/fonts", ref="main", download=True)

References:
    The project README covers installation, advanced usage, and contribution
    guidelines in greater depth.

"""

from torchfont.sample import GlyphSample

__all__ = ["GlyphSample"]
