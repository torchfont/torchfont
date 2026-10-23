use skia_safe::{Path, PathBuilder, PathFillType, PathVerb};

use kurbo::{BezPath, PathEl, Point};

pub(super) const SKIA_SCALE: f32 = 16_384.0;

pub(super) fn build_skia_path(elements: &[PathEl]) -> Path {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    for &element in elements {
        push_skia_element(&mut builder, element);
    }
    builder.detach()
}

fn push_skia_element(builder: &mut PathBuilder, element: PathEl) {
    match element {
        PathEl::MoveTo(point) => {
            builder.move_to((point.x as f32, point.y as f32));
        }
        PathEl::LineTo(point) => {
            builder.line_to((point.x as f32, point.y as f32));
        }
        PathEl::QuadTo(control, end) => {
            builder.quad_to(
                (control.x as f32, control.y as f32),
                (end.x as f32, end.y as f32),
            );
        }
        PathEl::CurveTo(control0, control1, end) => {
            builder.cubic_to(
                (control0.x as f32, control0.y as f32),
                (control1.x as f32, control1.y as f32),
                (end.x as f32, end.y as f32),
            );
        }
        PathEl::ClosePath => {
            builder.close();
        }
    };
}

pub(super) fn outline_from_path(path: &Path) -> Option<BezPath> {
    let mut outline = BezPath::new();
    let mut start = None;
    let mut elements: Vec<PathEl> = Vec::new();

    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => {
                commit_subpath(&mut outline, &mut start, &mut elements, false);
                start = Some(point(points[0]));
            }
            PathVerb::Line => elements.push(PathEl::LineTo(point(points[1]))),
            PathVerb::Quad => elements.push(PathEl::QuadTo(point(points[1]), point(points[2]))),
            PathVerb::Cubic => elements.push(PathEl::CurveTo(
                point(points[1]),
                point(points[2]),
                point(points[3]),
            )),
            PathVerb::Close => commit_subpath(&mut outline, &mut start, &mut elements, true),
            PathVerb::Conic => return None,
        }
    }
    commit_subpath(&mut outline, &mut start, &mut elements, false);

    Some(outline)
}

fn commit_subpath(
    outline: &mut BezPath,
    start: &mut Option<Point>,
    elements: &mut Vec<PathEl>,
    closed: bool,
) {
    if let Some(start) = start.take()
        && !elements.is_empty()
    {
        outline.move_to(start);
        outline.extend(elements.drain(..));
        if closed {
            outline.close_path();
        }
    }
}

fn point(point: skia_safe::Point) -> Point {
    Point::new(point.x.into(), point.y.into())
}
