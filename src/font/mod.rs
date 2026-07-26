mod data;
mod extract;
mod location;
mod registered_axes;

pub(crate) use data::{map_font, parse_font_ref};
pub(crate) use extract::extract_glyph_outline;
pub(crate) use location::{AxisInfo, Location, axis_info, canonicalize_location, default_location};
pub(crate) use registered_axes::{RegisteredAxisValues, registered_axis_values};
