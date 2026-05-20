mod entry;
mod extract;
pub(crate) mod glyph;
mod reader;
pub(crate) mod table;

pub(crate) use entry::FontEntry;
pub(crate) use extract::extract_glyph_outline;
