use crate::outline::{Outline, PathElement, Point, Subpath};

// Absolute tolerance for quadratic approximation (normalized coords ≈ 1 font-unit in 1000 UPM).
const TOLERANCE: f32 = 1e-3;
// Keep this aligned with fonttools.cu2qu.MAX_N.
const MAX_N: usize = 100;

type Cubic = (Point, Point, Point, Point);

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
                    let p3 = end;
                    for (qcp, end) in cubic_to_quads(prev, control0, control1, p3)? {
                        elements.push(PathElement::QuadTo { control: qcp, end });
                    }
                    prev = p3;
                }
                other => {
                    elements.push(other);
                    prev = other.end();
                }
            }
        }
        subpaths.push(Subpath::new(subpath.start(), elements, subpath.is_closed()));
    }
    Ok(outline.with_subpaths(subpaths))
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
                    midpoint(spline[i + 1], spline[i + 2])
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
    let mut d1 = Point::zero();
    spline.push(p0);
    spline.push(next_q1);

    for i in 1..=n {
        let (_c0, c1, c2, c3) = cubics[i - 1];
        let q0 = q2;
        let q1 = next_q1;
        if i < n {
            next_q1 = cubic_approx_control(i as f32 / (n - 1) as f32, cubics[i]);
            spline.push(next_q1);
            q2 = midpoint(q1, next_q1);
        } else {
            q2 = c3;
        }

        let d0 = d1;
        d1 = sub(q2, c3);
        let e1 = sub(lerp(q0, q1, 2.0 / 3.0), c1);
        let e2 = sub(lerp(q2, q1, 2.0 / 3.0), c2);
        if norm(d1) > TOLERANCE || !cubic_farthest_fit_inside(d0, e1, e2, d1, TOLERANCE) {
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
        .unwrap_or_else(|| midpoint(lerp(p0, p1, 1.5), lerp(p3, p2, 1.5)));
    let c1 = lerp(p0, q1, 2.0 / 3.0);
    let c2 = lerp(p3, q1, 2.0 / 3.0);
    cubic_farthest_fit_inside(
        Point::zero(),
        sub(c1, p1),
        sub(c2, p2),
        Point::zero(),
        TOLERANCE,
    )
    .then_some(q1)
}

fn cubic_approx_control(t: f32, (p0, p1, p2, p3): Cubic) -> Point {
    let a = lerp(p0, p1, 1.5);
    let b = lerp(p3, p2, 1.5);
    lerp(a, b, t)
}

fn line_intersection(a: Point, b: Point, c: Point, d: Point) -> Option<Point> {
    let ab = sub(b, a);
    let cd = sub(d, c);
    let den = cross(ab, cd);
    if den.abs() < 1e-15 {
        return (b == c && (a == b || c == d)).then_some(b);
    }
    let h = cross(sub(c, a), ab) / den;
    Some(lerp(c, d, h))
}

fn split_cubic_into_n(p0: Point, p1: Point, p2: Point, p3: Point, n: usize) -> Vec<Cubic> {
    if n == 1 {
        return vec![(p0, p1, p2, p3)];
    }
    let mut result = Vec::with_capacity(n);
    let mut current = (p0, p1, p2, p3);
    for k in 0..n - 1 {
        let (left, right) = split_cubic_at(
            current.0,
            current.1,
            current.2,
            current.3,
            1.0 / (n - k) as f32,
        );
        result.push(left);
        current = right;
    }
    result.push(current);
    result
}

fn split_cubic_at(p0: Point, p1: Point, p2: Point, p3: Point, t: f32) -> (Cubic, Cubic) {
    let q0 = lerp(p0, p1, t);
    let q1 = lerp(p1, p2, t);
    let q2 = lerp(p2, p3, t);
    let r0 = lerp(q0, q1, t);
    let r1 = lerp(q1, q2, t);
    let s = lerp(r0, r1, t);
    ((p0, q0, r0, s), (s, r1, q2, p3))
}

fn cubic_farthest_fit_inside(p0: Point, p1: Point, p2: Point, p3: Point, tolerance: f32) -> bool {
    if norm(p2) <= tolerance && norm(p1) <= tolerance {
        return true;
    }
    let mid = Point::new(
        (p0.x + 3.0 * (p1.x + p2.x) + p3.x) * 0.125,
        (p0.y + 3.0 * (p1.y + p2.y) + p3.y) * 0.125,
    );
    if norm(mid) > tolerance {
        return false;
    }
    let deriv3 = Point::new(
        (p3.x + p2.x - p1.x - p0.x) * 0.125,
        (p3.y + p2.y - p1.y - p0.y) * 0.125,
    );
    cubic_farthest_fit_inside(p0, midpoint(p0, p1), sub(mid, deriv3), mid, tolerance)
        && cubic_farthest_fit_inside(mid, add(mid, deriv3), midpoint(p2, p3), p3, tolerance)
}

fn lerp(a: Point, b: Point, t: f32) -> Point {
    a.lerp(b, t)
}
fn midpoint(a: Point, b: Point) -> Point {
    a.midpoint(b)
}
fn sub(a: Point, b: Point) -> Point {
    a.sub(b)
}
fn add(a: Point, b: Point) -> Point {
    a.add(b)
}
fn cross(a: Point, b: Point) -> f32 {
    a.cross(b)
}
fn norm(point: Point) -> f32 {
    point.norm()
}
