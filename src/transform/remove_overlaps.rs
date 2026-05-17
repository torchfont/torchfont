use skia_safe::{Path, PathBuilder, PathFillType, PathVerb};

use crate::outline::ElementType;

pub(crate) fn remove_overlaps(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let path = build_path(types, coords);
    path.simplify().map_or_else(
        || outline_from_input(types, coords),
        |simplified| outline_from_path(&simplified),
    )
}

fn build_path(types: &[i64], coords: &[f32]) -> Path {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    for (&element_type, values) in types.iter().zip(coords.chunks_exact(6)) {
        match element_type {
            v if v == ElementType::MoveTo as i64 => {
                builder.move_to((values[4], values[5]));
            }
            v if v == ElementType::LineTo as i64 => {
                builder.line_to((values[4], values[5]));
            }
            v if v == ElementType::QuadTo as i64 => {
                builder.quad_to((values[0], values[1]), (values[4], values[5]));
            }
            v if v == ElementType::CurveTo as i64 => {
                builder.cubic_to(
                    (values[0], values[1]),
                    (values[2], values[3]),
                    (values[4], values[5]),
                );
            }
            v if v == ElementType::Close as i64 => {
                builder.close();
            }
            _ => break,
        };
    }
    builder.detach()
}

fn outline_from_input(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let len = types
        .iter()
        .position(|&element_type| element_type == ElementType::End as i64)
        .map_or(types.len(), |idx| idx + 1);
    (types[..len].to_vec(), coords[..len * 6].to_vec())
}

fn outline_from_path(path: &Path) -> (Vec<i64>, Vec<f32>) {
    let mut types = Vec::with_capacity(path.verbs().len() + 1);
    let mut coords = Vec::with_capacity((path.verbs().len() + 1) * 6);

    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => {
                push_endpoint(&mut types, &mut coords, ElementType::MoveTo, points[0])
            }
            PathVerb::Line => {
                push_endpoint(&mut types, &mut coords, ElementType::LineTo, points[1])
            }
            PathVerb::Quad => push(
                &mut types,
                &mut coords,
                ElementType::QuadTo,
                [points[1].x, points[1].y, 0.0, 0.0, points[2].x, points[2].y],
            ),
            PathVerb::Cubic => push(
                &mut types,
                &mut coords,
                ElementType::CurveTo,
                [
                    points[1].x,
                    points[1].y,
                    points[2].x,
                    points[2].y,
                    points[3].x,
                    points[3].y,
                ],
            ),
            PathVerb::Close => push(&mut types, &mut coords, ElementType::Close, [0.0; 6]),
            PathVerb::Conic => unreachable!("PathOps simplify should not emit conic segments"),
        }
    }
    push(&mut types, &mut coords, ElementType::End, [0.0; 6]);
    (types, coords)
}

fn push_endpoint(
    types: &mut Vec<i64>,
    coords: &mut Vec<f32>,
    element_type: ElementType,
    point: skia_safe::Point,
) {
    push(
        types,
        coords,
        element_type,
        [0.0, 0.0, 0.0, 0.0, point.x, point.y],
    );
}

fn push(types: &mut Vec<i64>, coords: &mut Vec<f32>, element_type: ElementType, values: [f32; 6]) {
    types.push(element_type as i64);
    coords.extend_from_slice(&values);
}
