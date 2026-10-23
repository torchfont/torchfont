use kurbo::{BezPath, CubicBez, PathEl, PathSeg, Point, QuadBez};

use crate::outline::{subpath_is_closed, subpath_segments, subpath_start};
use crate::transform::curves::{TOLERANCE, cubic_farthest_fit_inside};

pub(crate) fn cubic_to_quad(outline: &BezPath) -> Option<BezPath> {
    let mut result = BezPath::new();

    for subpath in outline.subpaths() {
        let start = subpath_start(subpath);
        result.move_to(start);
        for segment in subpath_segments(subpath) {
            if let PathSeg::Cubic(cubic) = segment {
                append_cubic_as_quads(&mut result, cubic)?;
            } else {
                result.push(segment.as_path_el());
            }
        }
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    Some(result)
}

fn append_cubic_as_quads(result: &mut BezPath, cubic: CubicBez) -> Option<()> {
    if !cubic.is_finite() {
        return None;
    }
    if let Some(control) = degree_reduced_control(cubic) {
        result.quad_to(control, cubic.p3);
        return Some(());
    }
    let spline = cubic.approx_spline(TOLERANCE)?;
    result.extend(
        spline
            .to_quads()
            .map(|quad| PathEl::QuadTo(quad.p1, quad.p2)),
    );
    Some(())
}

fn degree_reduced_control(cubic: CubicBez) -> Option<Point> {
    let CubicBez { p0, p1, p2, p3 } = cubic;
    let from_start = p0.lerp(p1, 1.5);
    let from_end = p3.lerp(p2, 1.5);
    let control = from_start.midpoint(from_end);
    let raised = QuadBez::new(p0, control, p3).raise();
    let deviation = CubicBez::new(
        Point::ZERO,
        raised.p1 - p1.to_vec2(),
        raised.p2 - p2.to_vec2(),
        Point::ZERO,
    );
    cubic_farthest_fit_inside(deviation, TOLERANCE).then_some(control)
}
