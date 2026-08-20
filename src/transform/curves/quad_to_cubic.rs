use kurbo::QuadBez;

use crate::outline::{
    BezPath, PathEl, path_element_end, subpath_elements, subpath_is_closed, subpath_start,
};

pub(crate) fn quad_to_cubic(outline: &BezPath) -> BezPath {
    let mut result = BezPath::new();
    for subpath in outline.subpaths() {
        let start = subpath_start(subpath);
        result.move_to(start);
        let mut prev = start;
        result.extend(subpath_elements(subpath).iter().map(|&el| {
            let converted = if let PathEl::QuadTo(control, end) = el {
                let cubic = QuadBez::new(prev, control, end).raise();
                PathEl::CurveTo(cubic.p1, cubic.p2, cubic.p3)
            } else {
                el
            };
            prev = path_element_end(converted);
            converted
        }));
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    result
}
