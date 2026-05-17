use crate::outline::ElementType;

// Absolute tolerance for quadratic approximation (normalized coords ≈ 1 font-unit in 1000 UPM).
const TOLERANCE: f32 = 1e-3;
// Keep this aligned with fonttools.cu2qu.MAX_N.
const MAX_N: usize = 100;

type Pt = [f32; 2];
type Cubic = (Pt, Pt, Pt, Pt);

pub(crate) enum CubicToQuadError {
    ApproximationFailed,
}

pub(crate) fn cubic_to_quad(
    types: &[i64],
    coords: &[f32],
) -> Result<(Vec<i64>, Vec<f32>), CubicToQuadError> {
    let n = types.len();
    let mut out_types: Vec<i64> = Vec::with_capacity(n);
    let mut out_coords: Vec<f32> = Vec::with_capacity(n * 6);
    let mut prev = [0f32; 2];

    for (i, &ty) in types.iter().enumerate() {
        let c = &coords[i * 6..(i + 1) * 6];
        if ty == ElementType::CurveTo as i64 {
            let p3 = [c[4], c[5]];
            for (qcp, end) in cubic_to_quads(prev, [c[0], c[1]], [c[2], c[3]], p3)? {
                out_types.push(ElementType::QuadTo as i64);
                out_coords.extend_from_slice(&[qcp[0], qcp[1], 0.0, 0.0, end[0], end[1]]);
            }
            prev = p3;
        } else {
            out_types.push(ty);
            out_coords.extend_from_slice(c);
            if ty == ElementType::MoveTo as i64
                || ty == ElementType::LineTo as i64
                || ty == ElementType::QuadTo as i64
            {
                prev = [c[4], c[5]];
            }
        }
    }
    Ok((out_types, out_coords))
}

// Port of fonttools.cu2qu's all_quadratic=True path.  The returned pairs encode
// the quadratic spline as explicit path elements, with implied on-curves materialized
// at midpoints between adjacent off-curves.
fn cubic_to_quads(p0: Pt, p1: Pt, p2: Pt, p3: Pt) -> Result<Vec<(Pt, Pt)>, CubicToQuadError> {
    for n in 1..=MAX_N {
        if let Some(spline) = cubic_approx_spline(p0, p1, p2, p3, n) {
            let mut out = Vec::with_capacity(n);
            for i in 0..n {
                let end = if i + 1 == n {
                    p3
                } else {
                    midpt(spline[i + 1], spline[i + 2])
                };
                out.push((spline[i + 1], end));
            }
            return Ok(out);
        }
    }
    Err(CubicToQuadError::ApproximationFailed)
}

fn cubic_approx_spline(p0: Pt, p1: Pt, p2: Pt, p3: Pt, n: usize) -> Option<Vec<Pt>> {
    if n == 1 {
        let q1 = cubic_approx_quadratic(p0, p1, p2, p3)?;
        return Some(vec![p0, q1, p3]);
    }

    let cubics = split_cubic_into_n(p0, p1, p2, p3, n);
    let mut spline = Vec::with_capacity(n + 2);
    let mut next_q1 = cubic_approx_control(0.0, cubics[0]);
    let mut q2 = p0;
    let mut d1 = [0.0, 0.0];
    spline.push(p0);
    spline.push(next_q1);

    for i in 1..=n {
        let (_c0, c1, c2, c3) = cubics[i - 1];
        let q0 = q2;
        let q1 = next_q1;
        if i < n {
            next_q1 = cubic_approx_control(i as f32 / (n - 1) as f32, cubics[i]);
            spline.push(next_q1);
            q2 = midpt(q1, next_q1);
        } else {
            q2 = c3;
        }

        let d0 = d1;
        d1 = sub2(q2, c3);
        let e1 = sub2(lerp2(q0, q1, 2.0 / 3.0), c1);
        let e2 = sub2(lerp2(q2, q1, 2.0 / 3.0), c2);
        if dist0(d1) > TOLERANCE || !cubic_farthest_fit_inside(d0, e1, e2, d1, TOLERANCE) {
            return None;
        }
    }
    spline.push(p3);
    Some(spline)
}

fn cubic_approx_quadratic(p0: Pt, p1: Pt, p2: Pt, p3: Pt) -> Option<Pt> {
    // Most reducible cubics recover their quadratic control from the endpoint
    // tangent intersection. When both tangents are parallel (notably straight
    // degree-elevated quadratics), the intersection is at infinity, so recover
    // the same control from each cubic handle and let the fit test below decide.
    let q1 = line_intersection(p0, p1, p2, p3)
        .unwrap_or_else(|| midpt(lerp2(p0, p1, 1.5), lerp2(p3, p2, 1.5)));
    let c1 = lerp2(p0, q1, 2.0 / 3.0);
    let c2 = lerp2(p3, q1, 2.0 / 3.0);
    cubic_farthest_fit_inside(
        [0.0, 0.0],
        sub2(c1, p1),
        sub2(c2, p2),
        [0.0, 0.0],
        TOLERANCE,
    )
    .then_some(q1)
}

fn cubic_approx_control(t: f32, (p0, p1, p2, p3): Cubic) -> Pt {
    let a = lerp2(p0, p1, 1.5);
    let b = lerp2(p3, p2, 1.5);
    lerp2(a, b, t)
}

fn line_intersection(a: Pt, b: Pt, c: Pt, d: Pt) -> Option<Pt> {
    let ab = sub2(b, a);
    let cd = sub2(d, c);
    let den = cross(ab, cd);
    if den.abs() < 1e-15 {
        return (b == c && (a == b || c == d)).then_some(b);
    }
    let h = cross(sub2(c, a), ab) / den;
    Some(lerp2(c, d, h))
}

fn split_cubic_into_n(p0: Pt, p1: Pt, p2: Pt, p3: Pt, n: usize) -> Vec<Cubic> {
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

fn split_cubic_at(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: f32) -> (Cubic, Cubic) {
    let q0 = lerp2(p0, p1, t);
    let q1 = lerp2(p1, p2, t);
    let q2 = lerp2(p2, p3, t);
    let r0 = lerp2(q0, q1, t);
    let r1 = lerp2(q1, q2, t);
    let s = lerp2(r0, r1, t);
    ((p0, q0, r0, s), (s, r1, q2, p3))
}

fn cubic_farthest_fit_inside(p0: Pt, p1: Pt, p2: Pt, p3: Pt, tolerance: f32) -> bool {
    if dist0(p2) <= tolerance && dist0(p1) <= tolerance {
        return true;
    }
    let mid = [
        (p0[0] + 3.0 * (p1[0] + p2[0]) + p3[0]) * 0.125,
        (p0[1] + 3.0 * (p1[1] + p2[1]) + p3[1]) * 0.125,
    ];
    if dist0(mid) > tolerance {
        return false;
    }
    let deriv3 = [
        (p3[0] + p2[0] - p1[0] - p0[0]) * 0.125,
        (p3[1] + p2[1] - p1[1] - p0[1]) * 0.125,
    ];
    cubic_farthest_fit_inside(p0, midpt(p0, p1), sub2(mid, deriv3), mid, tolerance)
        && cubic_farthest_fit_inside(mid, add2(mid, deriv3), midpt(p2, p3), p3, tolerance)
}

fn lerp2(a: Pt, b: Pt, t: f32) -> Pt {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}
fn midpt(a: Pt, b: Pt) -> Pt {
    [(a[0] + b[0]) * 0.5, (a[1] + b[1]) * 0.5]
}
fn sub2(a: Pt, b: Pt) -> Pt {
    [a[0] - b[0], a[1] - b[1]]
}
fn add2(a: Pt, b: Pt) -> Pt {
    [a[0] + b[0], a[1] + b[1]]
}
fn cross(a: Pt, b: Pt) -> f32 {
    a[0] * b[1] - a[1] * b[0]
}
fn dist0(a: Pt) -> f32 {
    if !a[0].is_finite() || !a[1].is_finite() {
        return f32::INFINITY;
    }
    (a[0] * a[0] + a[1] * a[1]).sqrt()
}
