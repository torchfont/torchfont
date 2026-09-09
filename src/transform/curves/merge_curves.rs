use kurbo::{CubicBez, ParamCurve, QuadBez};
use smallvec::{SmallVec, smallvec};

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
                        curve_junction_is_mergeable,
                        try_merge_cubics_n,
                    ),
                    PathEl::QuadTo(..) => try_merge_run(
                        seg_start,
                        elements,
                        i,
                        curve_junction_is_mergeable,
                        try_merge_quads_n,
                    ),
                    PathEl::LineTo(_) => try_merge_run(
                        seg_start,
                        elements,
                        i,
                        |_, e| matches!(e, PathEl::LineTo(_)),
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
    can_join: impl Fn(PathEl, PathEl) -> bool,
    try_merge: fn(Point, &[PathEl]) -> Option<PathEl>,
) -> (PathEl, usize) {
    let mut run_end = i + 1;
    while run_end < elements.len() && can_join(elements[run_end - 1], elements[run_end]) {
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

fn curve_junction_is_mergeable(previous: PathEl, current: PathEl) -> bool {
    let (end_tan, start_tan) = match (previous, current) {
        (PathEl::QuadTo(h, end), PathEl::QuadTo(next_h, _))
        | (PathEl::CurveTo(_, h, end), PathEl::CurveTo(next_h, _, _)) => (end - h, next_h - end),
        _ => return false,
    };
    tangent_ratio(end_tan, start_tan).is_some()
}

fn tangent_ratio(end_tan: Vec2, start_tan: Vec2) -> Option<f64> {
    let len_end = end_tan.hypot();
    let len_start = start_tan.hypot();
    if len_end < 1e-10 {
        return None;
    }
    if len_start > 1e-10 {
        if end_tan.cross(start_tan).abs() > TOLERANCE * len_end * len_start {
            return None;
        }
        if end_tan.dot(start_tan) < 0.0 {
            return None;
        }
    }
    Some(len_start / len_end)
}

// Reconstruct normalized split parameters from cumulative tangent-length ratios
// at each junction: ratio_k = |start_tan_k| / |end_tan_{k-1}|.
fn compute_split_ts(
    n: usize,
    junction_tangents: impl Fn(usize) -> (Vec2, Vec2),
) -> Option<SmallVec<[f64; 8]>> {
    let mut prod_ratio = 1.0_f64;
    let mut sum_ratio = 1.0_f64;
    let mut ts_unnorm: SmallVec<[f64; 8]> = smallvec![1.0_f64];

    for k in 1..n {
        let (end_tan, start_tan) = junction_tangents(k);
        let ratio = tangent_ratio(end_tan, start_tan)?;
        prod_ratio *= ratio;
        sum_ratio += prod_ratio;
        ts_unnorm.push(sum_ratio);
    }

    ts_unnorm.pop();
    Some(ts_unnorm.iter().map(|&t| t / sum_ratio).collect())
}

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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pruned_search_matches_exhaustive_search() {
        let mut state = 42_u64;
        for _ in 0..128 {
            let mut elements = Vec::new();
            let mut start = Point::ZERO;
            for _ in 0..64 {
                state = state.wrapping_mul(6364136223846793005).wrapping_add(1);
                let x = ((state >> 32) % 7) as f64 - 3.0;
                let y = ((state >> 40) % 7) as f64 - 3.0;
                let end = start + Vec2::new(x, y);
                match (state >> 48) % 3 {
                    0 => elements.push(PathEl::LineTo(end)),
                    1 => {
                        let curve = QuadBez::new(start, start + Vec2::new(y, x), end);
                        for range in [0.0..0.5, 0.5..1.0] {
                            let piece = curve.subsegment(range);
                            elements.push(PathEl::QuadTo(piece.p1, piece.p2));
                        }
                    }
                    _ => {
                        let curve = CubicBez::new(
                            start,
                            start + Vec2::new(y, x),
                            end - Vec2::new(x, y),
                            end,
                        );
                        for range in [0.0..0.5, 0.5..1.0] {
                            let piece = curve.subsegment(range);
                            elements.push(PathEl::CurveTo(piece.p1, piece.p2, piece.p3));
                        }
                    }
                }
                start = end;
            }

            let mut expected = Vec::new();
            let mut i = 0;
            let mut start = Point::ZERO;
            while i < elements.len() {
                let merge = match elements[i] {
                    PathEl::LineTo(_) => try_merge_lines_n,
                    PathEl::QuadTo(..) => try_merge_quads_n,
                    PathEl::CurveTo(..) => try_merge_cubics_n,
                    _ => unreachable!(),
                };
                let (element, len) = try_merge_run(
                    start,
                    &elements,
                    i,
                    |a, b| std::mem::discriminant(&a) == std::mem::discriminant(&b),
                    merge,
                );
                expected.push(element);
                start = path_element_end(element);
                i += len;
            }
            assert_eq!(merge_subpath_elements(Point::ZERO, &elements), expected);
        }
    }
}
