mod entry;
mod extract;
pub(crate) mod glyph;
mod reader;
pub(crate) mod table;
mod variation;

pub(crate) use entry::FontEntry;
pub(crate) use extract::extract_glyph_outline;
pub(crate) use variation::VariationInstantiation;
