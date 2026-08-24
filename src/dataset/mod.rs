//! Dataset discovery and deterministic sample indexing.

mod discovered_font;
mod discovery;
mod index;
mod parallel;

pub(crate) use discovered_font::{DiscoveredCodepoints, DiscoveredGlyphs};
pub(crate) use discovery::{canonicalize_root, discover_font_files};
pub(crate) use index::{CodepointIndex, GlyphIndex};
pub(crate) use parallel::build_from_files;
