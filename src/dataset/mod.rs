//! Dataset discovery and deterministic sample indexing.

mod classes;
mod counts;
mod discovered_font;
mod discovery;
mod fixed;
mod targets;
mod variable;

pub(crate) use discovered_font::DiscoveredFont;
pub(crate) use discovery::{canonicalize_root, discover_font_files};
pub(crate) use fixed::{FixedFontEntry, FixedIndex};
pub(crate) use variable::{VariableFontEntry, VariableIndex};

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum IndexOverflow {
    SampleCount,
    StyleCount,
}
