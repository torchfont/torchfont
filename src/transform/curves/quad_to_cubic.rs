use kurbo::{BezPath, PathSeg};

use crate::outline::{subpath_is_closed, subpath_segments, subpath_start};

pub(crate) fn quad_to_cubic(outline: &BezPath) -> BezPath {
    let mut result = BezPath::new();
    for subpath in outline.subpaths() {
        let start = subpath_start(subpath);
        result.move_to(start);
        result.extend(subpath_segments(subpath).map(|segment| {
            if let PathSeg::Quad(quad) = segment {
                PathSeg::Cubic(quad.raise()).as_path_el()
            } else {
                segment.as_path_el()
            }
        }));
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    result
}
