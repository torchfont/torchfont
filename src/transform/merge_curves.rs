use crate::outline::{Outline, PathElement, Point, Subpath};

// Absolute tolerance for merge validation (normalized coords ≈ 1 font-unit in 1000 UPM).
const TOLERANCE: f32 = 1e-3;

type Cubic = (Point, Point, Point, Point);
type Quad = (Point, Point, Point);

pub(crate) fn merge_curves(outline: &Outline) -> Outline {
    let subpaths = outline
        .subpaths()
        .iter()
        .map(|subpath| {
            let elements = merge_subpath_elements(subpath.start(), subpath.elements().to_vec());
            Subpath::new(subpath.start(), elements, subpath.is_closed())
        })
        .collect();
    Outline::new(subpaths)
}

fn merge_subpath_elements(start: Point, elements: Vec<PathElement>) -> Vec<PathElement> {
    let n = elements.len();
    let mut result = Vec::with_capacity(n);
    let mut result_starts = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        let element = elements[i];
        let seg_start = result
            .last()
            .map_or(start, |element: &PathElement| element.end());

        match element {
            PathElement::CurveTo { .. } => {
                let mut run_end = i + 1;
                while run_end < n && matches!(elements[run_end], PathElement::CurveTo { .. }) {
                    run_end += 1;
                }
                let run_len = run_end - i;
                let mut merged = None;
                for merge_len in (2..=run_len).rev() {
                    if let Some(element) =
                        try_merge_cubics_n(seg_start, &elements[i..i + merge_len])
                    {
                        merged = Some((element, merge_len));
                        break;
                    }
                }
                if let Some((element, len)) = merged {
                    result.push(element);
                    result_starts.push(seg_start);
                    i += len;
                } else {
                    result.push(element);
                    result_starts.push(seg_start);
                    i += 1;
                }
            }
            PathElement::QuadTo { .. } => {
                let mut run_end = i + 1;
                while run_end < n && matches!(elements[run_end], PathElement::QuadTo { .. }) {
                    run_end += 1;
                }
                let run_len = run_end - i;
                let mut merged = None;
                for merge_len in (2..=run_len).rev() {
                    if let Some(element) = try_merge_quads_n(seg_start, &elements[i..i + merge_len])
                    {
                        merged = Some((element, merge_len));
                        break;
                    }
                }
                if let Some((element, len)) = merged {
                    result.push(element);
                    result_starts.push(seg_start);
                    i += len;
                } else {
                    result.push(element);
                    result_starts.push(seg_start);
                    i += 1;
                }
            }
            PathElement::LineTo(end) => {
                let merged = if let (Some(PathElement::LineTo(last_end)), Some(&last_start)) =
                    (result.last().copied(), result_starts.last())
                {
                    can_merge_lines(last_start, last_end, end).then_some(PathElement::LineTo(end))
                } else {
                    None
                };
                if let Some(element) = merged {
                    let saved_start = *result_starts.last().unwrap();
                    result.pop();
                    result_starts.pop();
                    result.push(element);
                    result_starts.push(saved_start);
                } else {
                    result.push(element);
                    result_starts.push(seg_start);
                }
                i += 1;
            }
        }
    }
    result
}

fn quad_points(element: PathElement) -> (Point, Point) {
    match element {
        PathElement::QuadTo { control, end } => (control, end),
        _ => unreachable!("quadratic run contains only quadratic elements"),
    }
}

fn cubic_points(element: PathElement) -> (Point, Point, Point) {
    match element {
        PathElement::CurveTo {
            control0,
            control1,
            end,
        } => (control0, control1, end),
        _ => unreachable!("cubic run contains only cubic elements"),
    }
}

// Attempt to merge n consecutive quadratic segments into one.
//
// The split parameters can be recovered from adjacent tangent-length ratios in
// the same way as cubics. A parent quadratic then has only one outer control
// point to reconstruct and validate.
fn try_merge_quads_n(p0: Point, segs: &[PathElement]) -> Option<PathElement> {
    let n = segs.len();
    debug_assert!(n >= 2);

    let mut prod_ratio = 1.0_f32;
    let mut sum_ratio = 1.0_f32;
    let mut ts_unnorm = vec![1.0_f32];

    for k in 1..n {
        let (prev_h, prev_end) = quad_points(segs[k - 1]);
        let (curr_h, _curr_end) = quad_points(segs[k]);

        let end_tan = prev_end - prev_h;
        let start_tan = curr_h - prev_end;
        let len_end = end_tan.norm();
        let len_start = start_tan.norm();
        if len_end < 1e-10 {
            return None;
        }
        if len_start > 1e-10 {
            if end_tan.cross(start_tan).abs() > TOLERANCE * len_end * len_start {
                return None;
            }
            if end_tan.x * start_tan.x + end_tan.y * start_tan.y < 0.0 {
                return None;
            }
        }

        let ratio = len_start / len_end;
        prod_ratio *= ratio;
        sum_ratio += prod_ratio;
        ts_unnorm.push(sum_ratio);
    }

    ts_unnorm.pop();
    let ts: Vec<f32> = ts_unnorm.iter().map(|&t| t / sum_ratio).collect();
    let t1 = ts[0];
    if !(1e-6..=1.0 - 1e-6).contains(&t1) {
        return None;
    }

    let (first_h, _first_end) = quad_points(segs[0]);
    let p1 = p0.lerp(first_h, 1.0 / t1);
    let (_last_h, p2) = quad_points(segs[n - 1]);

    if !validate_quad_merge(p0, p1, p2, segs, &ts) {
        return None;
    }

    Some(PathElement::QuadTo {
        control: p1,
        end: p2,
    })
}

fn validate_quad_merge(p0: Point, p1: Point, p2: Point, segs: &[PathElement], ts: &[f32]) -> bool {
    let pieces = split_quad_at_ts(p0, p1, p2, ts);
    for (k, (_rp0, rp1, rp2)) in pieces.iter().enumerate() {
        let (orig_h, orig_end) = quad_points(segs[k]);
        if (*rp1 - orig_h).norm() > TOLERANCE || (*rp2 - orig_end).norm() > TOLERANCE {
            return false;
        }
    }
    true
}

fn split_quad_at_ts(p0: Point, p1: Point, p2: Point, ts: &[f32]) -> Vec<Quad> {
    let mut pieces = Vec::with_capacity(ts.len() + 1);
    let mut current = (p0, p1, p2);
    let mut t_prev = 0.0_f32;
    for &t in ts {
        let remaining = 1.0 - t_prev;
        if remaining < 1e-10 {
            return pieces;
        }
        let t_rel = (t - t_prev) / remaining;
        let (lp0, lp1, lp2, rp0, rp1, rp2) =
            split_quad_at_t(current.0, current.1, current.2, t_rel);
        pieces.push((lp0, lp1, lp2));
        current = (rp0, rp1, rp2);
        t_prev = t;
    }
    pieces.push(current);
    pieces
}

fn split_quad_at_t(
    p0: Point,
    p1: Point,
    p2: Point,
    t: f32,
) -> (Point, Point, Point, Point, Point, Point) {
    let q1 = p0.lerp(p1, t);
    let q2 = p1.lerp(p2, t);
    let s = q1.lerp(q2, t);
    (p0, q1, s, s, q2, p2)
}

// Attempt to merge n consecutive cubic segments into one.
//
// Uses the fonttools qu2cu approach: reconstruct t-parameters from cumulative
// ratios of adjacent junction tangent lengths, then recover the outer control
// points P1/P2. Validity is confirmed by re-splitting and measuring curve error.
fn try_merge_cubics_n(p0: Point, segs: &[PathElement]) -> Option<PathElement> {
    let n = segs.len();
    debug_assert!(n >= 2);

    // Compute cumulative tangent-length ratios at each junction.
    // ratio_k = |start_tan_k| / |end_tan_{k-1}|
    // ts_unnorm accumulates partial sums; the last entry is discarded (= total).
    let mut prod_ratio = 1.0_f32;
    let mut sum_ratio = 1.0_f32;
    let mut ts_unnorm = vec![1.0_f32];

    for k in 1..n {
        let (_prev_h1, prev_h2, prev_end) = cubic_points(segs[k - 1]);
        let (curr_h1, _curr_h2, _curr_end) = cubic_points(segs[k]);

        let end_tan = prev_end - prev_h2;
        let start_tan = curr_h1 - prev_end;

        let len_end = end_tan.norm();
        let len_start = start_tan.norm();

        if len_end < 1e-10 {
            return None;
        }

        // Tangents at the junction must be parallel and in the same direction.
        if len_start > 1e-10 {
            if end_tan.cross(start_tan).abs() > TOLERANCE * len_end * len_start {
                return None;
            }
            if end_tan.x * start_tan.x + end_tan.y * start_tan.y < 0.0 {
                return None; // anti-parallel (cusp)
            }
        }

        let ratio = len_start / len_end;
        prod_ratio *= ratio;
        sum_ratio += prod_ratio;
        ts_unnorm.push(sum_ratio);
    }

    // Normalize: ts has n-1 elements, ts[0] = t1 (first junction), ts[n-2] = t_{n-1} (last).
    ts_unnorm.pop();
    let ts: Vec<f32> = ts_unnorm.iter().map(|&t| t / sum_ratio).collect();

    let t1 = ts[0];
    let t_last = *ts.last().unwrap();

    if !(1e-6..=1.0 - 1e-6).contains(&t1) || !(1e-6..=1.0 - 1e-6).contains(&t_last) {
        return None;
    }

    let (first_h1, _first_h2, _first_end) = cubic_points(segs[0]);
    let (_last_h1, last_h2, p3) = cubic_points(segs[n - 1]);

    // Recover outer control points from the split relationship:
    //   first_h1 = lerp(P0, P1, t1)  →  P1 = P0 + (first_h1 − P0) / t1
    //   last_h2  = lerp(P2, P3, t_last)  →  P2 = P3 + (last_h2 − P3) / (1 − t_last)
    let p1 = p0.lerp(first_h1, 1.0 / t1);
    let p2 = p3.lerp(last_h2, 1.0 / (1.0 - t_last));

    if !validate_cubic_merge(p0, p1, p2, p3, segs, &ts) {
        return None;
    }

    Some(PathElement::CurveTo {
        control0: p1,
        control1: p2,
        end: p3,
    })
}

fn validate_cubic_merge(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    segs: &[PathElement],
    ts: &[f32],
) -> bool {
    let pieces = split_cubic_at_ts(p0, p1, p2, p3, ts);
    let mut prev_end = p0;

    for (k, (rp0, rp1, rp2, rp3)) in pieces.iter().enumerate() {
        let (orig_h1, orig_h2, orig_end) = cubic_points(segs[k]);

        if (*rp3 - orig_end).norm() > TOLERANCE {
            return false;
        }

        // Check that the difference cubic lies within TOLERANCE of the origin.
        let d0 = *rp0 - prev_end;
        let d1 = *rp1 - orig_h1;
        let d2 = *rp2 - orig_h2;
        let d3 = *rp3 - orig_end;

        if !cubic_farthest_fit_inside(d0, d1, d2, d3, TOLERANCE) {
            return false;
        }

        prev_end = orig_end;
    }

    true
}

// Split cubic (P0,P1,P2,P3) at each t in ts (ascending), returning n+1 pieces.
// Each subsequent split uses the reparametrized t relative to the remaining curve.
fn split_cubic_at_ts(p0: Point, p1: Point, p2: Point, p3: Point, ts: &[f32]) -> Vec<Cubic> {
    let mut pieces = Vec::with_capacity(ts.len() + 1);
    let mut current = (p0, p1, p2, p3);
    let mut t_prev = 0.0_f32;

    for &t in ts {
        let remaining = 1.0 - t_prev;
        if remaining < 1e-10 {
            return pieces;
        }
        let t_rel = (t - t_prev) / remaining;
        let (lp0, lp1, lp2, lp3, rp0, rp1, rp2, rp3) =
            split_cubic_at_t(current.0, current.1, current.2, current.3, t_rel);
        pieces.push((lp0, lp1, lp2, lp3));
        current = (rp0, rp1, rp2, rp3);
        t_prev = t;
    }
    pieces.push(current);
    pieces
}

fn split_cubic_at_t(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    t: f32,
) -> (Point, Point, Point, Point, Point, Point, Point, Point) {
    let q1 = p0.lerp(p1, t);
    let q2 = p1.lerp(p2, t);
    let q3 = p2.lerp(p3, t);
    let r1 = q1.lerp(q2, t);
    let r2 = q2.lerp(q3, t);
    let s = r1.lerp(r2, t);
    (p0, q1, r1, s, s, r2, q3, p3)
}

// Recursive check: does the cubic (as a displacement field relative to the origin)
// lie entirely within `tolerance` of the origin?  Ported from fonttools qu2cu.
fn cubic_farthest_fit_inside(p0: Point, p1: Point, p2: Point, p3: Point, tolerance: f32) -> bool {
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

    let p01 = p0.lerp(p1, 0.5);
    let p23 = p2.lerp(p3, 0.5);

    cubic_farthest_fit_inside(p0, p01, mid - deriv3, mid, tolerance)
        && cubic_farthest_fit_inside(mid, mid + deriv3, p23, p3, tolerance)
}

fn can_merge_lines(start: Point, middle: Point, end: Point) -> bool {
    let (dx1, dy1) = (middle.x - start.x, middle.y - start.y);
    let (dx2, dy2) = (end.x - middle.x, end.y - middle.y);

    let len1_sq = dx1 * dx1 + dy1 * dy1;
    let len2_sq = dx2 * dx2 + dy2 * dy2;

    if len1_sq < 1e-12 || len2_sq < 1e-12 {
        return true;
    }

    let total_dx = end.x - start.x;
    let total_dy = end.y - start.y;
    let total_len = total_dx.hypot(total_dy);
    if total_len < 1e-6 {
        return false;
    }

    // Measure the actual geometric deviation of the join from the merged line,
    // not just its angle. The rest of the module interprets TOLERANCE as an
    // absolute distance in normalized coordinates.
    let distance = (dx1 * total_dy - dy1 * total_dx).abs() / total_len;
    if distance > TOLERANCE {
        return false;
    }

    dx1 * dx2 + dy1 * dy2 >= 0.0
}
