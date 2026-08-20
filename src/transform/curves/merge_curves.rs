use kurbo::{CubicBez, ParamCurve, QuadBez};

use crate::outline::{
    BezPath, PathEl, Point, Vec2, path_element_end, subpath_elements, subpath_is_closed,
    subpath_start,
};
use crate::transform::curves::{TOLERANCE, cubic_farthest_fit_inside, finite_hypot};

pub(crate) fn merge_curves(outline: &BezPath) -> BezPath {
    let mut result = BezPath::new();
    for subpath in outline.subpaths() {
        let start = subpath_start(subpath);
        result.move_to(start);
        result.extend(merge_subpath_elements(start, subpath_elements(subpath)));
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    result
}

fn merge_subpath_elements(start: Point, elements: &[PathEl]) -> Vec<PathEl> {
    let n = elements.len();
    let mut result = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        let element = elements[i];
        let seg_start = result
            .last()
            .map_or(start, |e: &PathEl| path_element_end(*e));

        match element {
            PathEl::CurveTo(..) | PathEl::QuadTo(..) | PathEl::LineTo(_) => {
                let (merged, len) = match element {
                    PathEl::CurveTo(..) => try_merge_run(
                        seg_start,
                        elements,
                        i,
                        |e| matches!(e, PathEl::CurveTo(..)),
                        try_merge_cubics_n,
                    ),
                    PathEl::QuadTo(..) => try_merge_run(
                        seg_start,
                        elements,
                        i,
                        |e| matches!(e, PathEl::QuadTo(..)),
                        try_merge_quads_n,
                    ),
                    PathEl::LineTo(_) => try_merge_run(
                        seg_start,
                        elements,
                        i,
                        |e| matches!(e, PathEl::LineTo(_)),
                        try_merge_lines_n,
                    ),
                    PathEl::MoveTo(_) | PathEl::ClosePath => {
                        unreachable!("subpath elements contain only drawing elements")
                    }
                };
                result.push(merged);
                i += len;
            }
            PathEl::MoveTo(_) | PathEl::ClosePath => {
                unreachable!("subpath elements contain only drawing elements")
            }
        }
    }
    result
}

fn try_merge_run(
    seg_start: Point,
    elements: &[PathEl],
    i: usize,
    is_same: impl Fn(PathEl) -> bool,
    try_merge: fn(Point, &[PathEl]) -> Option<PathEl>,
) -> (PathEl, usize) {
    let mut run_end = i + 1;
    while run_end < elements.len() && is_same(elements[run_end]) {
        run_end += 1;
    }
    let run_len = run_end - i;
    (2..=run_len)
        .rev()
        .find_map(|len| try_merge(seg_start, &elements[i..i + len]).map(|e| (e, len)))
        .unwrap_or((elements[i], 1))
}

fn quad_points(element: PathEl) -> (Point, Point) {
    match element {
        PathEl::QuadTo(control, end) => (control, end),
        _ => unreachable!("quadratic run contains only quadratic elements"),
    }
}

fn cubic_points(element: PathEl) -> (Point, Point, Point) {
    match element {
        PathEl::CurveTo(control0, control1, end) => (control0, control1, end),
        _ => unreachable!("cubic run contains only cubic elements"),
    }
}

// Reconstruct normalized split parameters from cumulative tangent-length ratios
// at each junction. ratio_k = |start_tan_k| / |end_tan_{k-1}|; ts_unnorm
// accumulates partial sums and the last entry (= total) is discarded.
// Returns None if any junction tangent is degenerate or forms a cusp.
fn compute_split_ts(
    n: usize,
    junction_tangents: impl Fn(usize) -> (Vec2, Vec2),
) -> Option<Vec<f64>> {
    let mut prod_ratio = 1.0_f64;
    let mut sum_ratio = 1.0_f64;
    let mut ts_unnorm = vec![1.0_f64];

    for k in 1..n {
        let (end_tan, start_tan) = junction_tangents(k);
        let len_end = end_tan.hypot();
        let len_start = start_tan.hypot();
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
fn try_merge_quads_n(p0: Point, segs: &[PathEl]) -> Option<PathEl> {
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

    Some(PathEl::QuadTo(p1, p2))
}

fn validate_quad_merge(p0: Point, p1: Point, p2: Point, segs: &[PathEl], ts: &[f64]) -> bool {
    let pieces = split_quad_at_ts(p0, p1, p2, ts);
    for (piece, seg) in pieces.iter().zip(segs) {
        let (orig_h, orig_end) = quad_points(*seg);
        if finite_hypot(piece.p1 - orig_h) > TOLERANCE
            || finite_hypot(piece.p2 - orig_end) > TOLERANCE
        {
            return false;
        }
    }
    true
}

fn split_quad_at_ts(p0: Point, p1: Point, p2: Point, ts: &[f64]) -> Vec<QuadBez> {
    let mut pieces = Vec::with_capacity(ts.len() + 1);
    let mut current = QuadBez::new(p0, p1, p2);
    let mut t_prev = 0.0_f64;
    for &t in ts {
        let remaining = 1.0 - t_prev;
        if remaining < 1e-10 {
            return pieces;
        }
        let t_rel = (t - t_prev) / remaining;
        let left = current.subsegment(0.0..t_rel);
        let right = current.subsegment(t_rel..1.0);
        pieces.push(left);
        current = right;
        t_prev = t;
    }
    pieces.push(current);
    pieces
}

// Attempt to merge n consecutive cubic segments into one.
//
// Uses the fonttools qu2cu approach: reconstruct t-parameters from cumulative
// ratios of adjacent junction tangent lengths, then recover the outer control
// points P1/P2. Validity is confirmed by re-splitting and measuring curve error.
fn try_merge_cubics_n(p0: Point, segs: &[PathEl]) -> Option<PathEl> {
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

    Some(PathEl::CurveTo(p1, p2, p3))
}

fn validate_cubic_merge(
    p0: Point,
    p1: Point,
    p2: Point,
    p3: Point,
    segs: &[PathEl],
    ts: &[f64],
) -> bool {
    let pieces = split_cubic_at_ts(p0, p1, p2, p3, ts);
    let mut prev_end = p0;

    for (piece, seg) in pieces.iter().zip(segs) {
        let (orig_h1, orig_h2, orig_end) = cubic_points(*seg);

        if finite_hypot(piece.p3 - orig_end) > TOLERANCE {
            return false;
        }

        // Check that the difference cubic lies within TOLERANCE of the origin.
        let d0 = piece.p0 - prev_end;
        let d1 = piece.p1 - orig_h1;
        let d2 = piece.p2 - orig_h2;
        let d3 = piece.p3 - orig_end;

        if !cubic_farthest_fit_inside(d0, d1, d2, d3, TOLERANCE) {
            return false;
        }

        prev_end = orig_end;
    }

    true
}

// Split cubic (P0,P1,P2,P3) at each t in ts (ascending), returning n+1 pieces.
// Each subsequent split uses the reparametrized t relative to the remaining curve.
fn split_cubic_at_ts(p0: Point, p1: Point, p2: Point, p3: Point, ts: &[f64]) -> Vec<CubicBez> {
    let mut pieces = Vec::with_capacity(ts.len() + 1);
    let mut current = CubicBez::new(p0, p1, p2, p3);
    let mut t_prev = 0.0_f64;

    for &t in ts {
        let remaining = 1.0 - t_prev;
        if remaining < 1e-10 {
            return pieces;
        }
        let t_rel = (t - t_prev) / remaining;
        let left = current.subsegment(0.0..t_rel);
        let right = current.subsegment(t_rel..1.0);
        pieces.push(left);
        current = right;
        t_prev = t;
    }
    pieces.push(current);
    pieces
}

fn try_merge_lines_n(start: Point, segs: &[PathEl]) -> Option<PathEl> {
    let end = path_element_end(*segs.last()?);
    let total = end - start;
    if total.x == 0.0 && total.y == 0.0 {
        return segs
            .iter()
            .all(|seg| path_element_end(*seg) == start)
            .then_some(PathEl::LineTo(end));
    }

    let mut previous = start;
    for seg in segs {
        let point = path_element_end(*seg);
        let direction = point - previous;
        if !points_are_collinear(start, point, end) || direction.dot(total) < 0.0 {
            return None;
        }
        previous = point;
    }

    Some(PathEl::LineTo(end))
}

fn points_are_collinear(a: Point, b: Point, c: Point) -> bool {
    let ab = b - a;
    let ac = c - a;
    let cross = ab.cross(ac).abs();
    let product_scale = (ab.x * ac.y).abs() + (ab.y * ac.x).abs();

    // Public coordinates are f32, so permit the rounding already present at the
    // tensor boundary even though the internal geometry uses f64.
    cross <= 8.0 * f64::from(f32::EPSILON) * product_scale
}
