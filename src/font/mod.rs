mod entry;
mod extract;
mod variation;

pub(crate) use entry::{FontEntry, map_font, parse_font_ref};
pub(crate) use extract::extract_glyph_outline;
pub(crate) use variation::{
    axis_info, canonicalize_location, default_location, grid_location_count, grid_locations,
    named_locations,
};
