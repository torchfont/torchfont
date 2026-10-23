use kurbo::{BezPath, ParamCurve};

use crate::outline::{subpath_is_closed, subpath_segments, subpath_start};

pub(crate) fn random_split_segments(
    outline: &BezPath,
    selection_values: &[f32],
    position_values: &[f32],
    split_probability: f32,
    split_range: (f32, f32),
) -> BezPath {
    let mut value_index = 0;
    let mut result = BezPath::new();
    for subpath in outline.subpaths() {
        let subpath_start = subpath_start(subpath);
        result.move_to(subpath_start);
        for segment in subpath_segments(subpath) {
            if selection_values[value_index] < split_probability {
                let t =
                    split_range.0 + (split_range.1 - split_range.0) * position_values[value_index];
                let t = f64::from(t);
                result.extend(
                    [segment.subsegment(0.0..t), segment.subsegment(t..1.0)]
                        .map(|segment| segment.as_path_el()),
                );
            } else {
                result.push(segment.as_path_el());
            }
            value_index += 1;
        }
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    result
}
