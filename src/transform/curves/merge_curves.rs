use kurbo::{BezPath, CubicBez, Line, ParamCurve, PathSeg, Point, QuadBez, Vec2};
use smallvec::{SmallVec, smallvec};

use crate::outline::{subpath_is_closed, subpath_segments, subpath_start};
use crate::transform::curves::{TOLERANCE, cubic_farthest_fit_inside};

pub(crate) fn merge_curves(outline: &BezPath) -> BezPath {
    let mut result = BezPath::new();
    let mut segments = Vec::new();
    for subpath in outline.subpaths() {
        result.move_to(subpath_start(subpath));
        segments.clear();
        segments.extend(subpath_segments(subpath));
        for mut run in segments.chunk_by(junction_is_mergeable) {
            let merge = match run[0] {
                PathSeg::Line(_) => try_merge_lines_n,
                PathSeg::Quad(_) => try_merge_quads_n,
                PathSeg::Cubic(_) => try_merge_cubics_n,
            };
            while !run.is_empty() {
                let (merged, len) = (2..=run.len())
                    .rev()
                    .find_map(|len| merge(&run[..len]).map(|segment| (segment, len)))
                    .unwrap_or((run[0], 1));
                result.push(merged.as_path_el());
                run = &run[len..];
            }
        }
        if subpath_is_closed(subpath) {
            result.close_path();
        }
    }
    result
}

fn junction_tangents(previous: PathSeg, current: PathSeg) -> Option<(Vec2, Vec2)> {
    match (previous, current) {
        (PathSeg::Quad(previous), PathSeg::Quad(current)) => {
            Some((previous.p2 - previous.p1, current.p1 - current.p0))
        }
        (PathSeg::Cubic(previous), PathSeg::Cubic(current)) => {
            Some((previous.p3 - previous.p2, current.p1 - current.p0))
        }
        _ => None,
    }
}

fn junction_is_mergeable(previous: &PathSeg, current: &PathSeg) -> bool {
    match (previous, current) {
        (PathSeg::Line(_), PathSeg::Line(_)) => true,
        _ => junction_tangents(*previous, *current)
            .and_then(tangent_ratio)
            .is_some(),
    }
}

fn tangent_ratio((end_tan, start_tan): (Vec2, Vec2)) -> Option<f64> {
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

fn compute_split_ts(segments: &[PathSeg]) -> Option<SmallVec<[f64; 8]>> {
    let mut prod_ratio = 1.0_f64;
    let mut sum_ratio = 1.0_f64;
    let mut ts_unnorm: SmallVec<[f64; 8]> = smallvec![1.0_f64];
    for pair in segments.windows(2) {
        prod_ratio *= tangent_ratio(junction_tangents(pair[0], pair[1])?)?;
        sum_ratio += prod_ratio;
        ts_unnorm.push(sum_ratio);
    }
    ts_unnorm.pop();
    Some(ts_unnorm.iter().map(|&t| t / sum_ratio).collect())
}

fn try_merge_quads_n(segments: &[PathSeg]) -> Option<PathSeg> {
    let (PathSeg::Quad(first), PathSeg::Quad(last)) = (*segments.first()?, *segments.last()?)
    else {
        return None;
    };
    let ts = compute_split_ts(segments)?;
    let t1 = ts[0];
    if !(1e-6..=1.0 - 1e-6).contains(&t1) {
        return None;
    }
    let curve = QuadBez::new(first.p0, first.p0.lerp(first.p1, 1.0 / t1), last.p2);
    split_at_ts(curve, &ts)
        .iter()
        .zip(segments)
        .all(|(piece, segment)| {
            matches!(segment, PathSeg::Quad(original)
                if (piece.p1 - original.p1).hypot() <= TOLERANCE
                    && (piece.p2 - original.p2).hypot() <= TOLERANCE)
        })
        .then_some(curve.into())
}

fn try_merge_cubics_n(segments: &[PathSeg]) -> Option<PathSeg> {
    let (PathSeg::Cubic(first), PathSeg::Cubic(last)) = (*segments.first()?, *segments.last()?)
    else {
        return None;
    };
    let ts = compute_split_ts(segments)?;
    let t1 = ts[0];
    let t_last = *ts.last()?;
    if !(1e-6..=1.0 - 1e-6).contains(&t1) || !(1e-6..=1.0 - 1e-6).contains(&t_last) {
        return None;
    }
    let curve = CubicBez::new(
        first.p0,
        first.p0.lerp(first.p1, 1.0 / t1),
        last.p3.lerp(last.p2, 1.0 / (1.0 - t_last)),
        last.p3,
    );
    split_at_ts(curve, &ts)
        .iter()
        .zip(segments)
        .all(|(piece, segment)| {
            matches!(segment, PathSeg::Cubic(original)
            if (piece.p3 - original.p3).hypot() <= TOLERANCE
            && cubic_farthest_fit_inside(
                CubicBez::new(
                    piece.p0 - original.p0.to_vec2(),
                    piece.p1 - original.p1.to_vec2(),
                    piece.p2 - original.p2.to_vec2(),
                    piece.p3 - original.p3.to_vec2(),
                ),
                TOLERANCE,
            ))
        })
        .then_some(curve.into())
}

fn split_at_ts<C: ParamCurve>(mut curve: C, ts: &[f64]) -> Vec<C> {
    let mut pieces = Vec::with_capacity(ts.len() + 1);
    let mut t_prev = 0.0;
    for &t in ts {
        let remaining = 1.0 - t_prev;
        if remaining < 1e-10 {
            return pieces;
        }
        let t_rel = (t - t_prev) / remaining;
        pieces.push(curve.subsegment(0.0..t_rel));
        curve = curve.subsegment(t_rel..1.0);
        t_prev = t;
    }
    pieces.push(curve);
    pieces
}

fn try_merge_lines_n(segments: &[PathSeg]) -> Option<PathSeg> {
    let merged = Line::new(segments.first()?.start(), segments.last()?.end());
    let total = merged.p1 - merged.p0;
    if total == Vec2::ZERO {
        return segments
            .iter()
            .all(|segment| segment.end() == merged.p0)
            .then_some(merged.into());
    }
    segments
        .iter()
        .all(|segment| {
            points_are_collinear(merged.p0, segment.end(), merged.p1)
                && (segment.end() - segment.start()).dot(total) >= 0.0
        })
        .then_some(merged.into())
}

fn points_are_collinear(a: Point, b: Point, c: Point) -> bool {
    let ab = b - a;
    let ac = c - a;
    let cross = ab.cross(ac).abs();
    let product_scale = (ab.x * ac.y).abs() + (ab.y * ac.x).abs();

    cross <= 8.0 * f64::from(f32::EPSILON) * product_scale
}
