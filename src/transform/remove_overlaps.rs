use skia_safe::{Path, PathBuilder, PathFillType, PathVerb};

use crate::outline::Element;

pub(crate) fn remove_overlaps(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let path = build_path(types, coords);
    path.simplify().map_or_else(
        || elements_from_input(types, coords),
        |simplified| path_to_elements(&simplified),
    )
}

fn build_path(types: &[i64], coords: &[f32]) -> Path {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    for (&element, values) in types.iter().zip(coords.chunks_exact(6)) {
        match element {
            v if v == Element::MoveTo as i64 => {
                builder.move_to((values[4], values[5]));
            }
            v if v == Element::LineTo as i64 => {
                builder.line_to((values[4], values[5]));
            }
            v if v == Element::QuadTo as i64 => {
                builder.quad_to((values[0], values[1]), (values[4], values[5]));
            }
            v if v == Element::CurveTo as i64 => {
                builder.cubic_to(
                    (values[0], values[1]),
                    (values[2], values[3]),
                    (values[4], values[5]),
                );
            }
            v if v == Element::Close as i64 => {
                builder.close();
            }
            _ => break,
        };
    }
    builder.detach()
}

fn elements_from_input(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let len = types
        .iter()
        .position(|&element| element == Element::End as i64)
        .map_or(types.len(), |idx| idx + 1);
    (types[..len].to_vec(), coords[..len * 6].to_vec())
}

fn path_to_elements(path: &Path) -> (Vec<i64>, Vec<f32>) {
    let mut types = Vec::with_capacity(path.verbs().len() + 1);
    let mut coords = Vec::with_capacity((path.verbs().len() + 1) * 6);

    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => push_endpoint(&mut types, &mut coords, Element::MoveTo, points[0]),
            PathVerb::Line => push_endpoint(&mut types, &mut coords, Element::LineTo, points[1]),
            PathVerb::Quad => push(
                &mut types,
                &mut coords,
                Element::QuadTo,
                [points[1].x, points[1].y, 0.0, 0.0, points[2].x, points[2].y],
            ),
            PathVerb::Cubic => push(
                &mut types,
                &mut coords,
                Element::CurveTo,
                [
                    points[1].x,
                    points[1].y,
                    points[2].x,
                    points[2].y,
                    points[3].x,
                    points[3].y,
                ],
            ),
            PathVerb::Close => push(&mut types, &mut coords, Element::Close, [0.0; 6]),
            PathVerb::Conic => unreachable!("PathOps simplify should not emit conic segments"),
        }
    }
    push(&mut types, &mut coords, Element::End, [0.0; 6]);
    (types, coords)
}

fn push_endpoint(
    types: &mut Vec<i64>,
    coords: &mut Vec<f32>,
    element: Element,
    point: skia_safe::Point,
) {
    push(
        types,
        coords,
        element,
        [0.0, 0.0, 0.0, 0.0, point.x, point.y],
    );
}

fn push(types: &mut Vec<i64>, coords: &mut Vec<f32>, element: Element, values: [f32; 6]) {
    types.push(element as i64);
    coords.extend_from_slice(&values);
}
