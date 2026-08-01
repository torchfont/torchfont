use super::split_cubic_at;
use crate::outline::{Outline, PathElement, Point, Subpath};

pub(crate) fn random_split_segments(
    outline: &Outline,
    selection_values: &[f32],
    position_values: &[f32],
    split_probability: f32,
    split_range: (f32, f32),
) -> Outline {
    let segment_count: usize = outline
        .subpaths()
        .iter()
        .map(|subpath| subpath.elements().len())
        .sum();
    if segment_count == 0 {
        return outline.clone();
    }

    let selection_values = &selection_values[..segment_count];
    let position_values = &position_values[..segment_count];
    let mut value_index = 0;
    let subpaths = outline
        .subpaths()
        .iter()
        .map(|subpath| {
            let mut start = subpath.start();
            let mut elements = Vec::with_capacity(subpath.elements().len() * 2);
            for &element in subpath.elements() {
                if selection_values[value_index] < split_probability {
                    let t = split_range.0
                        + (split_range.1 - split_range.0) * position_values[value_index];
                    elements.extend(split_segment(start, element, t));
                } else {
                    elements.push(element);
                }
                start = element.end();
                value_index += 1;
            }
            Subpath::new(subpath.start(), elements, subpath.is_closed())
        })
        .collect();
    Outline::new(subpaths)
}

fn split_segment(start: Point, element: PathElement, t: f32) -> [PathElement; 2] {
    match element {
        PathElement::LineTo(end) => {
            let split = start.lerp(end, t);
            [PathElement::LineTo(split), PathElement::LineTo(end)]
        }
        PathElement::QuadTo { control, end } => {
            let control0 = start.lerp(control, t);
            let control1 = control.lerp(end, t);
            let split = control0.lerp(control1, t);
            [
                PathElement::QuadTo {
                    control: control0,
                    end: split,
                },
                PathElement::QuadTo {
                    control: control1,
                    end,
                },
            ]
        }
        PathElement::CurveTo {
            control0,
            control1,
            end,
        } => {
            let (left, right) = split_cubic_at(start, control0, control1, end, t);
            [
                PathElement::CurveTo {
                    control0: left.1,
                    control1: left.2,
                    end: left.3,
                },
                PathElement::CurveTo {
                    control0: right.1,
                    control1: right.2,
                    end: right.3,
                },
            ]
        }
    }
}
