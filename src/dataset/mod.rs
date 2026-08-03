//! Dataset discovery and deterministic sample indexing.

mod classes;
mod discovered_font;
mod discovery;
mod glyph;

pub(crate) use discovered_font::DiscoveredFont;
pub(crate) use discovery::{canonicalize_root, discover_font_files};
pub(crate) use glyph::{FontEntry, GlyphIndex};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IndexOverflow {
    SampleCount,
}
