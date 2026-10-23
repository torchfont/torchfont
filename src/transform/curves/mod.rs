pub(crate) mod cubic_to_quad;
pub(crate) mod merge_curves;
pub(crate) mod quad_to_cubic;
pub(crate) mod split_segments;

use kurbo::{CubicBez, ParamCurve};

pub(crate) const TOLERANCE: f64 = 1e-3;

pub(crate) fn cubic_farthest_fit_inside(curve: CubicBez, tolerance: f64) -> bool {
    if !curve.is_finite() {
        return false;
    }
    if curve.p2.to_vec2().hypot() <= tolerance && curve.p1.to_vec2().hypot() <= tolerance {
        return true;
    }
    let (left, right) = curve.subdivide();
    left.p3.to_vec2().hypot() <= tolerance
        && cubic_farthest_fit_inside(left, tolerance)
        && cubic_farthest_fit_inside(right, tolerance)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measures_curve_extent_not_control_points() {
        let curve = CubicBez::new((0.0, 0.0), (2.0, 0.0), (-2.0, 0.0), (0.0, 0.0));
        assert!(cubic_farthest_fit_inside(curve, 0.6));
        assert!(!cubic_farthest_fit_inside(curve, 0.5));
    }

    #[test]
    fn nan_displacement_rejects_instead_of_recursing_forever() {
        let curve = CubicBez::new((0.0, 0.0), (f64::NAN, 0.0), (0.0, 0.0), (0.0, 0.0));
        assert!(!cubic_farthest_fit_inside(curve, TOLERANCE));
    }
}
