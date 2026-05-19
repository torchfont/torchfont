use skia_safe::{Path, PathBuilder, PathFillType};

use crate::geom::{Bounds, BoundsPen, Outline, PathElement, Point};

pub(crate) mod remove_overlaps;
pub(crate) mod render_bitmap;

pub(crate) fn build_skia_path(
    outline: &Outline,
    track_bounds: bool,
) -> Option<(Path, Option<Bounds>)> {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    let mut bounds = track_bounds.then(BoundsPen::default);
    for subpath in outline.subpaths() {
        let start = subpath.start();
        if let Some(bounds) = &mut bounds {
            bounds.move_to(start);
        }
        builder.move_to((start.x, start.y));
        for element in subpath.elements() {
            if let Some(bounds) = &mut bounds {
                bounds.path_element(*element);
            }
            match *element {
                PathElement::LineTo(point) => builder.line_to((point.x, point.y)),
                PathElement::QuadTo { control, end } => {
                    builder.quad_to((control.x, control.y), (end.x, end.y))
                }
                PathElement::CurveTo {
                    control0,
                    control1,
                    end,
                } => builder.cubic_to(
                    (control0.x, control0.y),
                    (control1.x, control1.y),
                    (end.x, end.y),
                ),
            };
        }
        if subpath.is_closed() {
            if let Some(bounds) = &mut bounds {
                bounds.close();
            }
            builder.close();
        }
    }
    (!builder.is_empty()).then(|| (builder.detach(), bounds.and_then(BoundsPen::finish)))
}

// Absolute tolerance for curve operations (normalized coords ≈ 1 font-unit in 1000 UPM).
pub(crate) const TOLERANCE: f32 = 1e-3;

pub(crate) type Cubic = (Point, Point, Point, Point);

pub(crate) fn split_cubic_at(p0: Point, p1: Point, p2: Point, p3: Point, t: f32) -> (Cubic, Cubic) {
    let q0 = p0.lerp(p1, t);
    let q1 = p1.lerp(p2, t);
    let q2 = p2.lerp(p3, t);
    let r0 = q0.lerp(q1, t);
    let r1 = q1.lerp(q2, t);
    let s = r0.lerp(r1, t);
    ((p0, q0, r0, s), (s, r1, q2, p3))
}

// Recursive check: does the cubic (as a displacement field relative to the origin)
// lie entirely within `tolerance` of the origin? Ported from fonttools qu2cu.
pub(crate) fn cubic_farthest_fit_inside(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    tolerance: f32,
) -> bool {
    if p2.norm() <= tolerance && p1.norm() <= tolerance {
        return true;
    }
    let mid = Point::new(
        (p0.x + 3.0 * (p1.x + p2.x) + p3.x) * 0.125,
        (p0.y + 3.0 * (p1.y + p2.y) + p3.y) * 0.125,
    );
    if mid.norm() > tolerance {
        return false;
    }
    let deriv3 = Point::new(
        (p3.x + p2.x - p1.x - p0.x) * 0.125,
        (p3.y + p2.y - p1.y - p0.y) * 0.125,
    );
    cubic_farthest_fit_inside(p0, p0.midpoint(p1), mid - deriv3, mid, tolerance)
        && cubic_farthest_fit_inside(mid, mid + deriv3, p2.midpoint(p3), p3, tolerance)
}
