use kurbo::{CubicBez, cubics_to_quadratic_splines};

use crate::outline::{
    BezPath, PathEl, Point, Vec2, path_element_end, subpath_elements, subpath_is_closed,
    subpath_start,
};
use crate::transform::curves::{TOLERANCE, cubic_farthest_fit_inside};

pub(crate) enum CubicToQuadError {
    ApproximationFailed,
}

pub(crate) fn cubic_to_quad(outline: &BezPath) -> Result<BezPath, CubicToQuadError> {
    let mut result = BezPath::new();

    for subpath in outline.subpaths() {
        let start = subpath_start(subpath);
        result.move_to(start);
        let mut prev = start;
        for element in subpath_elements(subpath) {
            match *element {
                PathEl::CurveTo(control0, control1, end) => {
                    append_cubic_as_quads(&mut result, prev, control0, control1, end)?;
                    prev = end;
                }
                other => {
                    result.push(other);
                    prev = path_element_end(other);
                }
            }
        }
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    Ok(result)
}

// Port of fonttools.cu2qu's all_quadratic=True path.  The returned pairs encode
// the quadratic spline as explicit path elements, with implied on-curves materialized
// at midpoints between adjacent off-curves.
fn append_cubic_as_quads(
    result: &mut BezPath,
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
) -> Result<(), CubicToQuadError> {
    let cubic = CubicBez::new(p0, p1, p2, p3);
    // Kurbo's recursive fitting assumes finite coordinates.
    if !cubic.is_finite() {
        return Err(CubicToQuadError::ApproximationFailed);
    }
    // Preserve a single quadratic for degree-reduced cubics. A cubic that is
    // an exact degree elevation of a quadratic has tangent lines at p0/p3
    // that are exactly parallel, which Kurbo's crossing-point fit cannot
    // resolve (it only special-cases fully coincident controls), so it would
    // otherwise emit two segments for input this crate's own quad-to-cubic
    // conversion produces.
    if let Some(control) = degree_reduced_control(p0, p1, p2, p3) {
        result.quad_to(control, p3);
        return Ok(());
    }
    let spline = cubics_to_quadratic_splines(&[cubic], TOLERANCE)
        .ok_or(CubicToQuadError::ApproximationFailed)?
        .pop()
        .expect("one cubic produces one spline");
    result.extend(
        spline
            .to_quads()
            .map(|quad| PathEl::QuadTo(quad.p1, quad.p2)),
    );
    Ok(())
}

fn degree_reduced_control(p0: Point, p1: Point, p2: Point, p3: Point) -> Option<Point> {
    let from_start = p0.lerp(p1, 1.5);
    let from_end = p3.lerp(p2, 1.5);
    let control = from_start.midpoint(from_end);
    let cubic_control0 = p0.lerp(control, 2.0 / 3.0);
    let cubic_control1 = p3.lerp(control, 2.0 / 3.0);
    // Validate the round trip the same way merge_curves does: as a
    // difference cubic that must lie within tolerance everywhere, not just
    // at the handles.
    let d1 = cubic_control0 - p1;
    let d2 = cubic_control1 - p2;
    cubic_farthest_fit_inside(Vec2::ZERO, d1, d2, Vec2::ZERO, TOLERANCE).then_some(control)
}
