mod bounds;
mod encoding;
mod path;

pub(crate) use bounds::{Bounds, bounds_from_outline, bounds_from_subpath};
pub(crate) use encoding::{DecodeError, ElementType, decode, encode};
pub(crate) use kurbo::{BezPath, PathEl, Point, Vec2};
#[cfg(test)]
pub(crate) use path::outline_from_subpaths;
pub(crate) use path::{
    path_element_end, path_seg, subpath_elements, subpath_from_elements, subpath_is_closed,
    subpath_start,
};
