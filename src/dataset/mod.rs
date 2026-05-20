pub(crate) mod index;
mod io;

pub(crate) use crate::font::FontEntry;
pub(crate) use index::{DatasetIndex, load_entries_and_index};
pub(crate) use io::{canonicalize_root, discover_font_files};
