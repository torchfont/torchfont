pub(crate) mod entry;
pub(crate) mod index;
mod io;
mod reader;

pub(crate) use entry::FontEntry;
pub(crate) use index::{DatasetIndex, load_entries_and_index};
pub(crate) use io::{canonicalize_root, discover_font_files};
