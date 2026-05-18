use super::{Cubic, TOLERANCE, cubic_farthest_fit_inside, split_cubic_at};
use crate::outline::{Outline, PathElement, Point, Subpath};

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
        let seg_start = result.last().map_or(start, |e: &PathElement| e.end());

        match element {
            PathElement::CurveTo { .. } => {
                let (element, len) = try_merge_run(
                    seg_start,
                    &elements,
                    i,
                    n,
                    |e| matches!(e, PathElement::CurveTo { .. }),
                    try_merge_cubics_n,
                );
                result.push(element);
                result_starts.push(seg_start);
                i += len;
            }
            PathElement::QuadTo { .. } => {
                let (element, len) = try_merge_run(
                    seg_start,
                    &elements,
                    i,
                    n,
                    |e| matches!(e, PathElement::QuadTo { .. }),
                    try_merge_quads_n,
                );
                result.push(element);
                result_starts.push(seg_start);
                i += len;
            }
            PathElement::LineTo(end) => {
                if let (Some(PathElement::LineTo(last_end)), Some(&last_start)) =
                    (result.last().copied(), result_starts.last())
                    && can_merge_lines(last_start, last_end, end)
                {
                    result.pop();
                    result_starts.pop();
                    result.push(PathElement::LineTo(end));
                    result_starts.push(last_start);
                    i += 1;
                    continue;
                }
                result.push(element);
                result_starts.push(seg_start);
                i += 1;
            }
        }
    }
    result
}

fn try_merge_run(
    seg_start: Point,
    elements: &[PathElement],
    i: usize,
    n: usize,
    is_same: impl Fn(PathElement) -> bool,
    try_merge: fn(Point, &[PathElement]) -> Option<PathElement>,
) -> (PathElement, usize) {
    let mut run_end = i + 1;
    while run_end < n && is_same(elements[run_end]) {
        run_end += 1;
    }
    let run_len = run_end - i;
    (2..=run_len)
        .rev()
        .find_map(|len| try_merge(seg_start, &elements[i..i + len]).map(|e| (e, len)))
        .unwrap_or((elements[i], 1))
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

// Reconstruct normalized split parameters from cumulative tangent-length ratios
// at each junction. ratio_k = |start_tan_k| / |end_tan_{k-1}|; ts_unnorm
// accumulates partial sums and the last entry (= total) is discarded.
// Returns None if any junction tangent is degenerate or forms a cusp.
fn compute_split_ts(
    n: usize,
    junction_tangents: impl Fn(usize) -> (Point, Point),
) -> Option<Vec<f32>> {
    let mut prod_ratio = 1.0_f32;
    let mut sum_ratio = 1.0_f32;
    let mut ts_unnorm = vec![1.0_f32];

    for k in 1..n {
        let (end_tan, start_tan) = junction_tangents(k);
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
            if end_tan.dot(start_tan) < 0.0 {
                return None;
            }
        }
        let ratio = len_start / len_end;
        prod_ratio *= ratio;
        sum_ratio += prod_ratio;
        ts_unnorm.push(sum_ratio);
    }

    // ts has n-1 elements; ts[0] = t1 (first junction), ts[n-2] = t_{n-1} (last).
    ts_unnorm.pop();
    Some(ts_unnorm.iter().map(|&t| t / sum_ratio).collect())
}

// Attempt to merge n consecutive quadratic segments into one.
fn try_merge_quads_n(p0: Point, segs: &[PathElement]) -> Option<PathElement> {
    let n = segs.len();
    debug_assert!(n >= 2);

    let ts = compute_split_ts(n, |k| {
        let (prev_h, prev_end) = quad_points(segs[k - 1]);
        let (curr_h, _) = quad_points(segs[k]);
        (prev_end - prev_h, curr_h - prev_end)
    })?;

    let t1 = ts[0];
    if !(1e-6..=1.0 - 1e-6).contains(&t1) {
        return None;
    }

    let (first_h, _) = quad_points(segs[0]);
    let p1 = p0.lerp(first_h, 1.0 / t1);
    let (_, p2) = quad_points(segs[n - 1]);

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
    for ((_rp0, rp1, rp2), seg) in pieces.iter().zip(segs) {
        let (orig_h, orig_end) = quad_points(*seg);
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
        let (p0, p1, p2) = current;
        let (left, right) = split_quad_at_t(p0, p1, p2, t_rel);
        pieces.push(left);
        current = right;
        t_prev = t;
    }
    pieces.push(current);
    pieces
}

fn split_quad_at_t(p0: Point, p1: Point, p2: Point, t: f32) -> (Quad, Quad) {
    let q1 = p0.lerp(p1, t);
    let q2 = p1.lerp(p2, t);
    let s = q1.lerp(q2, t);
    ((p0, q1, s), (s, q2, p2))
}

// Attempt to merge n consecutive cubic segments into one.
//
// Uses the fonttools qu2cu approach: reconstruct t-parameters from cumulative
// ratios of adjacent junction tangent lengths, then recover the outer control
// points P1/P2. Validity is confirmed by re-splitting and measuring curve error.
fn try_merge_cubics_n(p0: Point, segs: &[PathElement]) -> Option<PathElement> {
    let n = segs.len();
    debug_assert!(n >= 2);

    let ts = compute_split_ts(n, |k| {
        let (_, prev_h2, prev_end) = cubic_points(segs[k - 1]);
        let (curr_h1, _, _) = cubic_points(segs[k]);
        (prev_end - prev_h2, curr_h1 - prev_end)
    })?;

    let t1 = ts[0];
    let t_last = *ts.last().unwrap();

    if !(1e-6..=1.0 - 1e-6).contains(&t1) || !(1e-6..=1.0 - 1e-6).contains(&t_last) {
        return None;
    }

    let (first_h1, _, _) = cubic_points(segs[0]);
    let (_, last_h2, p3) = cubic_points(segs[n - 1]);

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

    for ((rp0, rp1, rp2, rp3), seg) in pieces.iter().zip(segs) {
        let (orig_h1, orig_h2, orig_end) = cubic_points(*seg);

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
        let (p0, p1, p2, p3) = current;
        let (left, right) = split_cubic_at(p0, p1, p2, p3, t_rel);
        pieces.push(left);
        current = right;
        t_prev = t;
    }
    pieces.push(current);
    pieces
}

fn can_merge_lines(start: Point, middle: Point, end: Point) -> bool {
    let d1 = middle - start;
    let d2 = end - middle;

    let len1_sq = d1.dot(d1);
    let len2_sq = d2.dot(d2);

    if len1_sq < 1e-12 || len2_sq < 1e-12 {
        return true;
    }

    let total = end - start;
    let total_len = total.norm();
    if total_len < 1e-6 {
        return false;
    }

    // Measure the actual geometric deviation of the join from the merged line,
    // not just its angle. The rest of the module interprets TOLERANCE as an
    // absolute distance in normalized coordinates.
    if d1.cross(total).abs() / total_len > TOLERANCE {
        return false;
    }

    d1.dot(d2) >= 0.0
}
