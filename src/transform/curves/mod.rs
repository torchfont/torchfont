pub(crate) mod cubic_to_quad;
pub(crate) mod merge_curves;
pub(crate) mod quad_to_cubic;
pub(crate) mod split_segments;

use crate::outline::Vec2;

// Absolute tolerance in em units (≈ 1 font unit in a 1000-UPM font).
pub(crate) const TOLERANCE: f64 = 1e-3;

// Recursive check: does the cubic (as a displacement field relative to the origin)
// lie entirely within `tolerance` of the origin? Ported from fonttools qu2cu.
pub(crate) fn cubic_farthest_fit_inside(
    p0: Vec2,
    p1: Vec2,
    p2: Vec2,
    p3: Vec2,
    tolerance: f64,
) -> bool {
    if finite_hypot(p2) <= tolerance && finite_hypot(p1) <= tolerance {
        return true;
    }
    let mid = Vec2::new(
        (p0.x + 3.0 * (p1.x + p2.x) + p3.x) * 0.125,
        (p0.y + 3.0 * (p1.y + p2.y) + p3.y) * 0.125,
    );
    if finite_hypot(mid) > tolerance {
        return false;
    }
    let deriv3 = Vec2::new(
        (p3.x + p2.x - p1.x - p0.x) * 0.125,
        (p3.y + p2.y - p1.y - p0.y) * 0.125,
    );
    cubic_farthest_fit_inside(p0, (p0 + p1) * 0.5, mid - deriv3, mid, tolerance)
        && cubic_farthest_fit_inside(mid, mid + deriv3, (p2 + p3) * 0.5, p3, tolerance)
}

// A non-finite displacement (e.g. from a NaN coordinate) must compare as
// arbitrarily far rather than as neither near nor far: `NaN <= tolerance`
// and `NaN > tolerance` are both false, which would let corrupted input
// silently pass tolerance checks throughout this module instead of being
// rejected.
pub(crate) fn finite_hypot(v: Vec2) -> f64 {
    if v.is_finite() {
        v.hypot()
    } else {
        f64::INFINITY
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nan_displacement_rejects_instead_of_recursing_forever() {
        let nan = Vec2::new(f64::NAN, 0.0);
        let zero = Vec2::new(0.0, 0.0);
        assert!(!cubic_farthest_fit_inside(zero, nan, zero, zero, TOLERANCE));
    }
}
