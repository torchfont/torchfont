//! Dataset discovery and deterministic sample indexing.

mod discovered_font;
mod discovery;
mod index;

pub(crate) use discovered_font::DiscoveredFont;
pub(crate) use discovery::{canonicalize_root, discover_font_files};
pub(crate) use index::FontIndex;
