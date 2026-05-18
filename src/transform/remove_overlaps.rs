use skia_safe::{Path, PathVerb};

use crate::outline::{Outline, PathElement, Point, Subpath};

pub(crate) fn remove_overlaps(outline: &Outline) -> Outline {
    let Some((path, _)) = super::build_skia_path(outline, false) else {
        return outline.clone();
    };
    path.simplify().map_or_else(
        || outline.clone(),
        |simplified| {
            let simplified = outline_from_path(&simplified);
            outline.with_subpaths(simplified.subpaths().to_vec())
        },
    )
}

fn outline_from_path(path: &Path) -> Outline {
    let mut subpaths = Vec::new();
    let mut start = None;
    let mut elements = Vec::new();
    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => {
                if let Some(start) = start.replace(point(points[0])) {
                    subpaths.push(Subpath::new(start, std::mem::take(&mut elements), false));
                }
            }
            PathVerb::Line => elements.push(PathElement::LineTo(point(points[1]))),
            PathVerb::Quad => elements.push(PathElement::QuadTo {
                control: point(points[1]),
                end: point(points[2]),
            }),
            PathVerb::Cubic => elements.push(PathElement::CurveTo {
                control0: point(points[1]),
                control1: point(points[2]),
                end: point(points[3]),
            }),
            PathVerb::Close => {
                if let Some(start) = start.take() {
                    subpaths.push(Subpath::new(start, std::mem::take(&mut elements), true));
                }
            }
            PathVerb::Conic => unreachable!("PathOps simplify should not emit conic segments"),
        }
    }
    if let Some(start) = start {
        subpaths.push(Subpath::new(start, elements, false));
    }
    Outline::new(subpaths)
}

fn point(point: skia_safe::Point) -> Point {
    Point::new(point.x, point.y)
}
