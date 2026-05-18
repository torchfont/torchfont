use super::{Cubic, TOLERANCE, cubic_farthest_fit_inside, split_cubic_at};
use crate::outline::{Outline, PathElement, Point, Subpath};

// Keep this aligned with fonttools.cu2qu.MAX_N.
const MAX_N: usize = 100;

pub(crate) enum CubicToQuadError {
    ApproximationFailed,
}

pub(crate) fn cubic_to_quad(outline: &Outline) -> Result<Outline, CubicToQuadError> {
    let mut subpaths = Vec::with_capacity(outline.subpaths().len());

    for subpath in outline.subpaths() {
        let mut elements = Vec::with_capacity(subpath.elements().len());
        let mut prev = subpath.start();
        for element in subpath.elements() {
            match *element {
                PathElement::CurveTo {
                    control0,
                    control1,
                    end,
                } => {
                    for (qcp, quad_end) in cubic_to_quads(prev, control0, control1, end)? {
                        elements.push(PathElement::QuadTo {
                            control: qcp,
                            end: quad_end,
                        });
                    }
                    prev = end;
                }
                other => {
                    elements.push(other);
                    prev = other.end();
                }
            }
        }
        subpaths.push(Subpath::new(subpath.start(), elements, subpath.is_closed()));
    }
    Ok(Outline::new(subpaths))
}

// Port of fonttools.cu2qu's all_quadratic=True path.  The returned pairs encode
// the quadratic spline as explicit path elements, with implied on-curves materialized
// at midpoints between adjacent off-curves.
fn cubic_to_quads(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
) -> Result<Vec<(Point, Point)>, CubicToQuadError> {
    for n in 1..=MAX_N {
        if let Some(spline) = cubic_approx_spline(p0, p1, p2, p3, n) {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let end = if i + 1 == n {
                    p3
                } else {
                    spline[i + 1].midpoint(spline[i + 2])
                };
                out.push((spline[i + 1], end));
            }
            return Ok(out);
        }
    }
    Err(CubicToQuadError::ApproximationFailed)
}

fn cubic_approx_spline(p0: Point, p1: Point, p2: Point, p3: Point, n: usize) -> Option<Vec<Point>> {
    if n == 1 {
        let q1 = cubic_approx_quadratic(p0, p1, p2, p3)?;
        return Some(vec![p0, q1, p3]);
    }

    let cubics = split_cubic_into_n(p0, p1, p2, p3, n);
    let mut spline = Vec::with_capacity(n + 2);
    let mut next_q1 = cubic_approx_control(0.0, cubics[0]);
    let mut q2 = p0;
    let mut d1 = Point::default();
    spline.push(p0);
    spline.push(next_q1);

    for i in 1..=n {
        let (_c0, c1, c2, c3) = cubics[i - 1];
        let q0 = q2;
        let q1 = next_q1;
        if i < n {
            next_q1 = cubic_approx_control(i as f32 / (n - 1) as f32, cubics[i]);
            spline.push(next_q1);
            q2 = q1.midpoint(next_q1);
        } else {
            q2 = c3;
        }

        let d0 = d1;
        d1 = q2 - c3;
        let e1 = q0.lerp(q1, 2.0 / 3.0) - c1;
        let e2 = q2.lerp(q1, 2.0 / 3.0) - c2;
        if d1.norm() > TOLERANCE || !cubic_farthest_fit_inside(d0, e1, e2, d1, TOLERANCE) {
            return None;
        }
    }
    spline.push(p3);
    Some(spline)
}

fn cubic_approx_quadratic(p0: Point, p1: Point, p2: Point, p3: Point) -> Option<Point> {
    // Most reducible cubics recover their quadratic control from the endpoint
    // tangent intersection. When both tangents are parallel (notably straight
    // degree-elevated quadratics), the intersection is at infinity, so recover
    // the same control from each cubic handle and let the fit test below decide.
    let q1 = line_intersection(p0, p1, p2, p3)
        .unwrap_or_else(|| p0.lerp(p1, 1.5).midpoint(p3.lerp(p2, 1.5)));
    let c1 = p0.lerp(q1, 2.0 / 3.0);
    let c2 = p3.lerp(q1, 2.0 / 3.0);
    cubic_farthest_fit_inside(
        Point::default(),
        c1 - p1,
        c2 - p2,
        Point::default(),
        TOLERANCE,
    )
    .then_some(q1)
}

fn cubic_approx_control(t: f32, (p0, p1, p2, p3): Cubic) -> Point {
    let a = p0.lerp(p1, 1.5);
    let b = p3.lerp(p2, 1.5);
    a.lerp(b, t)
}

fn line_intersection(a: Point, b: Point, c: Point, d: Point) -> Option<Point> {
    let ab = b - a;
    let cd = d - c;
    let den = ab.cross(cd);
    if den.abs() < 1e-15 {
        return (b == c && (a == b || c == d)).then_some(b);
    }
    let h = (c - a).cross(ab) / den;
    Some(c.lerp(d, h))
}

fn split_cubic_into_n(p0: Point, p1: Point, p2: Point, p3: Point, n: usize) -> Vec<Cubic> {
    if n == 1 {
        return vec![(p0, p1, p2, p3)];
    }
    let mut result = Vec::with_capacity(n);
    let mut current = (p0, p1, p2, p3);
    for k in 0..n - 1 {
        let (p0, p1, p2, p3) = current;
        let (left, right) = split_cubic_at(p0, p1, p2, p3, 1.0 / (n - k) as f32);
        result.push(left);
        current = right;
    }
    result.push(current);
    result
}
