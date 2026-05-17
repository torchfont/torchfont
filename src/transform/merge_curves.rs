use crate::outline::ElementType;

// Absolute tolerance for merge validation (normalized coords ≈ 1 font-unit in 1000 UPM).
const TOLERANCE: f32 = 1e-3;

type Pt = [f32; 2];
type Cubic = (Pt, Pt, Pt, Pt);
type Quad = (Pt, Pt, Pt);

pub(crate) fn merge_curves(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    let n = types.len();
    let mut out_types: Vec<i64> = Vec::with_capacity(n);
    let mut out_coords: Vec<f32> = Vec::with_capacity(n * 6);

    let mut i = 0;
    while i < n {
        let ty = types[i];
        let c = arr6(coords, i);

        if ty == ElementType::MoveTo as i64 {
            let move_x = c[4];
            let move_y = c[5];
            i += 1;

            let mut elements: Vec<(i64, [f32; 6])> = Vec::new();
            while i < n {
                let st = types[i];
                if st == ElementType::Close as i64
                    || st == ElementType::End as i64
                    || st == ElementType::MoveTo as i64
                {
                    break;
                }
                elements.push((st, arr6(coords, i)));
                i += 1;
            }

            let merged = merge_subpath_elements(move_x, move_y, elements);

            out_types.push(ElementType::MoveTo as i64);
            out_coords.extend_from_slice(&c);
            for (mt, mc) in merged {
                out_types.push(mt);
                out_coords.extend_from_slice(&mc);
            }
        } else if ty == ElementType::End as i64 {
            out_types.push(ty);
            out_coords.extend_from_slice(&c);
            break;
        } else {
            out_types.push(ty);
            out_coords.extend_from_slice(&c);
            i += 1;
        }
    }

    (out_types, out_coords)
}

fn merge_subpath_elements(
    start_x: f32,
    start_y: f32,
    elements: Vec<(i64, [f32; 6])>,
) -> Vec<(i64, [f32; 6])> {
    let n = elements.len();
    let mut result: Vec<(i64, [f32; 6])> = Vec::with_capacity(n);
    let mut result_starts: Vec<[f32; 2]> = Vec::with_capacity(n);
    let mut i = 0;

    while i < n {
        let (ty, c) = elements[i];
        let seg_start = match result.last() {
            Some((_, lc)) => [lc[4], lc[5]],
            None => [start_x, start_y],
        };

        let line = ElementType::LineTo as i64;
        let quad = ElementType::QuadTo as i64;
        let cubic = ElementType::CurveTo as i64;

        if ty == cubic {
            let mut run_end = i + 1;
            while run_end < n && elements[run_end].0 == cubic {
                run_end += 1;
            }
            let run_len = run_end - i;

            let mut merged = None;
            for merge_len in (2..=run_len).rev() {
                if let Some(m) = try_merge_cubics_n(seg_start, &elements[i..i + merge_len]) {
                    merged = Some((m, merge_len));
                    break;
                }
            }

            if let Some((m, len)) = merged {
                result.push((cubic, m));
                result_starts.push(seg_start);
                i += len;
            } else {
                result.push((ty, c));
                result_starts.push(seg_start);
                i += 1;
            }
        } else if ty == quad {
            let mut run_end = i + 1;
            while run_end < n && elements[run_end].0 == quad {
                run_end += 1;
            }
            let run_len = run_end - i;

            let mut merged = None;
            for merge_len in (2..=run_len).rev() {
                if let Some(m) = try_merge_quads_n(seg_start, &elements[i..i + merge_len]) {
                    merged = Some((m, merge_len));
                    break;
                }
            }

            if let Some((m, len)) = merged {
                result.push((quad, m));
                result_starts.push(seg_start);
                i += len;
            } else {
                result.push((ty, c));
                result_starts.push(seg_start);
                i += 1;
            }
        } else if ty == line {
            let merged = if let (Some(&(last_ty, last_c)), Some(&ls)) =
                (result.last(), result_starts.last())
            {
                if last_ty == line {
                    let (ax, ay) = (last_c[4], last_c[5]);
                    let (bx, by) = (c[4], c[5]);
                    if can_merge_lines(ls[0], ls[1], ax, ay, bx, by) {
                        Some([0.0f32, 0.0, 0.0, 0.0, bx, by])
                    } else {
                        None
                    }
                } else {
                    None
                }
            } else {
                None
            };

            if let Some(new_c) = merged {
                let saved_start = *result_starts.last().unwrap();
                result.pop();
                result_starts.pop();
                result.push((line, new_c));
                result_starts.push(saved_start);
            } else {
                result.push((ty, c));
                result_starts.push(seg_start);
            }
            i += 1;
        } else {
            result.push((ty, c));
            result_starts.push(seg_start);
            i += 1;
        }
    }

    result
}

// Attempt to merge n consecutive quadratic segments into one.
//
// The split parameters can be recovered from adjacent tangent-length ratios in
// the same way as cubics. A parent quadratic then has only one outer control
// point to reconstruct and validate.
fn try_merge_quads_n(p0: Pt, segs: &[(i64, [f32; 6])]) -> Option<[f32; 6]> {
    let n = segs.len();
    debug_assert!(n >= 2);

    let mut prod_ratio = 1.0_f32;
    let mut sum_ratio = 1.0_f32;
    let mut ts_unnorm = vec![1.0_f32];

    for k in 1..n {
        let prev_c = &segs[k - 1].1;
        let curr_c = &segs[k].1;
        let prev_end = [prev_c[4], prev_c[5]];
        let prev_h = [prev_c[0], prev_c[1]];
        let curr_h = [curr_c[0], curr_c[1]];

        let end_tan = sub2(prev_end, prev_h);
        let start_tan = sub2(curr_h, prev_end);
        let len_end = hypot(end_tan[0], end_tan[1]);
        let len_start = hypot(start_tan[0], start_tan[1]);
        if len_end < 1e-10 {
            return None;
        }
        if len_start > 1e-10 {
            let cross = end_tan[0] * start_tan[1] - end_tan[1] * start_tan[0];
            if cross.abs() > TOLERANCE * len_end * len_start {
                return None;
            }
            if end_tan[0] * start_tan[0] + end_tan[1] * start_tan[1] < 0.0 {
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

    let first_h = [segs[0].1[0], segs[0].1[1]];
    let p1 = [
        p0[0] + (first_h[0] - p0[0]) / t1,
        p0[1] + (first_h[1] - p0[1]) / t1,
    ];
    let last_c = &segs[n - 1].1;
    let p2 = [last_c[4], last_c[5]];

    if !validate_quad_merge(p0, p1, p2, segs, &ts) {
        return None;
    }

    Some([p1[0], p1[1], 0.0, 0.0, p2[0], p2[1]])
}

fn validate_quad_merge(p0: Pt, p1: Pt, p2: Pt, segs: &[(i64, [f32; 6])], ts: &[f32]) -> bool {
    let pieces = split_quad_at_ts(p0, p1, p2, ts);
    for (k, (_rp0, rp1, rp2)) in pieces.iter().enumerate() {
        let orig_c = &segs[k].1;
        let orig_h = [orig_c[0], orig_c[1]];
        let orig_end = [orig_c[4], orig_c[5]];
        if hypot(rp1[0] - orig_h[0], rp1[1] - orig_h[1]) > TOLERANCE
            || hypot(rp2[0] - orig_end[0], rp2[1] - orig_end[1]) > TOLERANCE
        {
            return false;
        }
    }
    true
}

fn split_quad_at_ts(p0: Pt, p1: Pt, p2: Pt, ts: &[f32]) -> Vec<Quad> {
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

fn split_quad_at_t(p0: Pt, p1: Pt, p2: Pt, t: f32) -> (Pt, Pt, Pt, Pt, Pt, Pt) {
    let q1 = lerp2(p0, p1, t);
    let q2 = lerp2(p1, p2, t);
    let s = lerp2(q1, q2, t);
    (p0, q1, s, s, q2, p2)
}

// Attempt to merge n consecutive cubic segments into one.
//
// Uses the fonttools qu2cu approach: reconstruct t-parameters from cumulative
// ratios of adjacent junction tangent lengths, then recover the outer control
// points P1/P2. Validity is confirmed by re-splitting and measuring curve error.
fn try_merge_cubics_n(p0: [f32; 2], segs: &[(i64, [f32; 6])]) -> Option<[f32; 6]> {
    let n = segs.len();
    debug_assert!(n >= 2);

    // Compute cumulative tangent-length ratios at each junction.
    // ratio_k = |start_tan_k| / |end_tan_{k-1}|
    // ts_unnorm accumulates partial sums; the last entry is discarded (= total).
    let mut prod_ratio = 1.0_f32;
    let mut sum_ratio = 1.0_f32;
    let mut ts_unnorm = vec![1.0_f32];

    for k in 1..n {
        let prev_c = &segs[k - 1].1;
        let curr_c = &segs[k].1;

        let prev_end = [prev_c[4], prev_c[5]];
        let prev_h2 = [prev_c[2], prev_c[3]];
        let curr_h1 = [curr_c[0], curr_c[1]];

        let end_tan = sub2(prev_end, prev_h2);
        let start_tan = sub2(curr_h1, prev_end);

        let len_end = hypot(end_tan[0], end_tan[1]);
        let len_start = hypot(start_tan[0], start_tan[1]);

        if len_end < 1e-10 {
            return None;
        }

        // Tangents at the junction must be parallel and in the same direction.
        if len_start > 1e-10 {
            let cross = end_tan[0] * start_tan[1] - end_tan[1] * start_tan[0];
            if cross.abs() > TOLERANCE * len_end * len_start {
                return None;
            }
            if end_tan[0] * start_tan[0] + end_tan[1] * start_tan[1] < 0.0 {
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

    let first_c = &segs[0].1;
    let last_c = &segs[n - 1].1;
    let first_h1 = [first_c[0], first_c[1]];
    let last_h2 = [last_c[2], last_c[3]];
    let p3 = [last_c[4], last_c[5]];

    // Recover outer control points from the split relationship:
    //   first_h1 = lerp(P0, P1, t1)  →  P1 = P0 + (first_h1 − P0) / t1
    //   last_h2  = lerp(P2, P3, t_last)  →  P2 = P3 + (last_h2 − P3) / (1 − t_last)
    let p1 = [
        p0[0] + (first_h1[0] - p0[0]) / t1,
        p0[1] + (first_h1[1] - p0[1]) / t1,
    ];
    let p2 = [
        p3[0] + (last_h2[0] - p3[0]) / (1.0 - t_last),
        p3[1] + (last_h2[1] - p3[1]) / (1.0 - t_last),
    ];

    if !validate_cubic_merge(p0, p1, p2, p3, segs, &ts) {
        return None;
    }

    Some([p1[0], p1[1], p2[0], p2[1], p3[0], p3[1]])
}

fn validate_cubic_merge(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    segs: &[(i64, [f32; 6])],
    ts: &[f32],
) -> bool {
    let pieces = split_cubic_at_ts(p0, p1, p2, p3, ts);
    let mut prev_end = p0;

    for (k, (rp0, rp1, rp2, rp3)) in pieces.iter().enumerate() {
        let orig_c = &segs[k].1;
        let orig_h1 = [orig_c[0], orig_c[1]];
        let orig_h2 = [orig_c[2], orig_c[3]];
        let orig_end = [orig_c[4], orig_c[5]];

        if hypot(rp3[0] - orig_end[0], rp3[1] - orig_end[1]) > TOLERANCE {
            return false;
        }

        // Check that the difference cubic lies within TOLERANCE of the origin.
        let d0 = sub2(*rp0, prev_end);
        let d1 = sub2(*rp1, orig_h1);
        let d2 = sub2(*rp2, orig_h2);
        let d3 = sub2(*rp3, orig_end);

        if !cubic_farthest_fit_inside(d0, d1, d2, d3, TOLERANCE) {
            return false;
        }

        prev_end = orig_end;
    }

    true
}

// Split cubic (P0,P1,P2,P3) at each t in ts (ascending), returning n+1 pieces.
// Each subsequent split uses the reparametrized t relative to the remaining curve.
fn split_cubic_at_ts(p0: Pt, p1: Pt, p2: Pt, p3: Pt, ts: &[f32]) -> Vec<Cubic> {
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

fn split_cubic_at_t(p0: Pt, p1: Pt, p2: Pt, p3: Pt, t: f32) -> (Pt, Pt, Pt, Pt, Pt, Pt, Pt, Pt) {
    let q1 = lerp2(p0, p1, t);
    let q2 = lerp2(p1, p2, t);
    let q3 = lerp2(p2, p3, t);
    let r1 = lerp2(q1, q2, t);
    let r2 = lerp2(q2, q3, t);
    let s = lerp2(r1, r2, t);
    (p0, q1, r1, s, s, r2, q3, p3)
}

// Recursive check: does the cubic (as a displacement field relative to the origin)
// lie entirely within `tolerance` of the origin?  Ported from fonttools qu2cu.
fn cubic_farthest_fit_inside(
    p0: [f32; 2],
    p1: [f32; 2],
    p2: [f32; 2],
    p3: [f32; 2],
    tolerance: f32,
) -> bool {
    if hypot(p2[0], p2[1]) <= tolerance && hypot(p1[0], p1[1]) <= tolerance {
        return true;
    }

    let mid = [
        (p0[0] + 3.0 * (p1[0] + p2[0]) + p3[0]) * 0.125,
        (p0[1] + 3.0 * (p1[1] + p2[1]) + p3[1]) * 0.125,
    ];
    if hypot(mid[0], mid[1]) > tolerance {
        return false;
    }

    let deriv3 = [
        (p3[0] + p2[0] - p1[0] - p0[0]) * 0.125,
        (p3[1] + p2[1] - p1[1] - p0[1]) * 0.125,
    ];

    let p01 = lerp2(p0, p1, 0.5);
    let p23 = lerp2(p2, p3, 0.5);

    cubic_farthest_fit_inside(p0, p01, sub2(mid, deriv3), mid, tolerance)
        && cubic_farthest_fit_inside(mid, add2(mid, deriv3), p23, p3, tolerance)
}

fn can_merge_lines(px: f32, py: f32, ax: f32, ay: f32, bx: f32, by: f32) -> bool {
    let (dx1, dy1) = (ax - px, ay - py);
    let (dx2, dy2) = (bx - ax, by - ay);

    let len1_sq = dx1 * dx1 + dy1 * dy1;
    let len2_sq = dx2 * dx2 + dy2 * dy2;

    if len1_sq < 1e-12 || len2_sq < 1e-12 {
        return true;
    }

    let total_dx = bx - px;
    let total_dy = by - py;
    let total_len = hypot(total_dx, total_dy);
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

#[inline]
fn lerp2(a: [f32; 2], b: [f32; 2], t: f32) -> [f32; 2] {
    [a[0] + (b[0] - a[0]) * t, a[1] + (b[1] - a[1]) * t]
}

#[inline]
fn sub2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] - b[0], a[1] - b[1]]
}

#[inline]
fn add2(a: [f32; 2], b: [f32; 2]) -> [f32; 2] {
    [a[0] + b[0], a[1] + b[1]]
}

#[inline]
fn arr6(coords: &[f32], i: usize) -> [f32; 6] {
    let mut a = [0f32; 6];
    a.copy_from_slice(&coords[i * 6..(i + 1) * 6]);
    a
}

#[inline]
fn hypot(x: f32, y: f32) -> f32 {
    (x * x + y * y).sqrt()
}
