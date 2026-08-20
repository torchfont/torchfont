use kurbo::ParamCurve;

use crate::outline::{
    BezPath, PathEl, Point, path_element_end, path_seg, subpath_elements, subpath_is_closed,
    subpath_start,
};

pub(crate) fn random_split_segments(
    outline: &BezPath,
    selection_values: &[f32],
    position_values: &[f32],
    split_probability: f32,
    split_range: (f32, f32),
) -> BezPath {
    let segment_count: usize = outline
        .subpaths()
        .map(|subpath| subpath_elements(subpath).len())
        .sum();
    if segment_count == 0 {
        return outline.clone();
    }

    let selection_values = &selection_values[..segment_count];
    let position_values = &position_values[..segment_count];
    let mut value_index = 0;
    let mut result = BezPath::new();
    for subpath in outline.subpaths() {
        let subpath_start = subpath_start(subpath);
        result.move_to(subpath_start);
        let mut start = subpath_start;
        for &element in subpath_elements(subpath) {
            if selection_values[value_index] < split_probability {
                let t =
                    split_range.0 + (split_range.1 - split_range.0) * position_values[value_index];
                result.extend(split_segment(start, element, t));
            } else {
                result.push(element);
            }
            start = path_element_end(element);
            value_index += 1;
        }
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    result
}

fn split_segment(start: Point, element: PathEl, t: f32) -> [PathEl; 2] {
    let segment = path_seg(start, element);
    let t = t.into();
    [segment.subsegment(0.0..t), segment.subsegment(t..1.0)].map(|segment| segment.as_path_el())
}
