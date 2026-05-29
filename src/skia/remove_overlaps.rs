use crate::geom::reverse_subpath;
use skia_safe::{Path, PathBuilder, PathFillType, PathOp, PathVerb, Point as SkPoint, Rect};

use crate::geom::{Outline, PathElement, Point, Subpath};
use crate::skia::render_bitmap::{RenderMode, render_bitmap};

// Skia PathOps has numerical precision issues with sub-unit coordinates.
// Scale up before simplify and back down after to ensure reliable results.
const PATHOPS_SCALE: f32 = 16384.0;
const CANDIDATE_SCALES: [f32; 5] = [1.0, 1024.0, 4096.0, 16384.0, 65536.0];
const SCORE_BITMAP_SIZE: u32 = 64;

// Skia PathOps is super-linear in element count; skip very complex outlines.
const MAX_PATHOPS_ELEMENTS: usize = 500;

pub(crate) fn remove_overlaps(outline: &Outline) -> Outline {
    let total_elements: usize = outline.subpaths().iter().map(|s| s.elements().len()).sum();
    if total_elements > MAX_PATHOPS_ELEMENTS {
        return outline.clone();
    }

    let scaled = scale_outline(outline, PATHOPS_SCALE);
    if !needs_pathops(&scaled) {
        return outline.clone();
    }
    if !has_hard_overlap(outline) {
        return outline.clone();
    }

    fast_candidate(outline)
        .or_else(|| best_candidate(outline))
        .unwrap_or_else(|| outline.clone())
}

/// Try Union then Simplify at PATHOPS_SCALE; return the first that matches the
/// original winding bitmap.  Faster than best_candidate but may miss edge cases.
fn fast_candidate(outline: &Outline) -> Option<Outline> {
    let original = render_bitmap(
        outline,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    )
    .ok()?;

    for mode in [PathOpsMode::Union, PathOpsMode::Simplify] {
        let Some(candidate) =
            pathops_candidate(outline, PATHOPS_SCALE, mode, PathFillType::Winding)
        else {
            continue;
        };
        let Ok(candidate_bmp) = render_bitmap(
            &candidate,
            SCORE_BITMAP_SIZE,
            RenderMode::Fixed,
            PathFillType::Winding,
        ) else {
            continue;
        };
        if hard_diff(&original.data, &candidate_bmp.data) == 0 {
            return Some(candidate);
        }
    }
    None
}

fn has_hard_overlap(outline: &Outline) -> bool {
    let Ok(winding) = render_bitmap(
        outline,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    ) else {
        return true;
    };
    let Ok(even_odd) = render_bitmap(
        outline,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::EvenOdd,
    ) else {
        return true;
    };
    hard_diff(&winding.data, &even_odd.data) != 0
}

fn best_candidate(outline: &Outline) -> Option<Outline> {
    let original = render_bitmap(
        outline,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    )
    .ok()?;

    let mut best: Option<(CandidateScore, Outline)> = None;
    for (scale, mode, fill_type) in [
        (PATHOPS_SCALE, PathOpsMode::Union, PathFillType::Winding),
        (PATHOPS_SCALE, PathOpsMode::Simplify, PathFillType::Winding),
        (PATHOPS_SCALE, PathOpsMode::FoldUnion, PathFillType::Winding),
        (PATHOPS_SCALE, PathOpsMode::Union, PathFillType::EvenOdd),
        (4096.0, PathOpsMode::Union, PathFillType::Winding),
        (4096.0, PathOpsMode::Simplify, PathFillType::Winding),
        (4096.0, PathOpsMode::FoldUnion, PathFillType::Winding),
        (4096.0, PathOpsMode::Union, PathFillType::EvenOdd),
    ] {
        consider_candidate(&mut best, &original.data, outline, scale, mode, fill_type);
    }
    consider_reoriented_candidates(&mut best, &original.data, outline);

    let needs_full_search = best
        .as_ref()
        .is_none_or(|(score, _)| score.overlap_pixels != 0 || score.mismatch_pixels != 0);
    if needs_full_search {
        for scale in CANDIDATE_SCALES {
            for mode in [
                PathOpsMode::Union,
                PathOpsMode::Simplify,
                PathOpsMode::FoldUnion,
            ] {
                for fill_type in [PathFillType::Winding, PathFillType::EvenOdd] {
                    consider_candidate(&mut best, &original.data, outline, scale, mode, fill_type);
                }
            }
        }
    }

    best.map(|(_, candidate)| candidate)
}

fn consider_reoriented_candidates(
    best: &mut Option<(CandidateScore, Outline)>,
    original_winding: &[u8],
    outline: &Outline,
) {
    let n = outline.subpaths().len();
    if !(2..=8).contains(&n) {
        return;
    }
    for mask in 1usize..(1usize << n) {
        let subpaths = outline
            .subpaths()
            .iter()
            .enumerate()
            .map(|(idx, subpath)| {
                if (mask & (1usize << idx)) == 0 {
                    subpath.clone()
                } else {
                    reverse_subpath(subpath)
                }
            })
            .collect();
        let candidate = Outline::new(subpaths);
        let Some(score) = score_candidate(original_winding, &candidate) else {
            continue;
        };
        if best
            .as_ref()
            .is_none_or(|(best_score, _)| score < *best_score)
        {
            *best = Some((score, candidate));
        }
    }
}

fn consider_candidate(
    best: &mut Option<(CandidateScore, Outline)>,
    original_winding: &[u8],
    outline: &Outline,
    scale: f32,
    mode: PathOpsMode,
    fill_type: PathFillType,
) {
    let Some(candidate) = pathops_candidate(outline, scale, mode, fill_type) else {
        return;
    };
    let Some(score) = score_candidate(original_winding, &candidate) else {
        return;
    };
    if best
        .as_ref()
        .is_none_or(|(best_score, _)| score < *best_score)
    {
        *best = Some((score, candidate));
    }
}

#[derive(Clone, Copy)]
enum PathOpsMode {
    Union,
    Simplify,
    FoldUnion,
}

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct CandidateScore {
    overlap_pixels: usize,
    mismatch_pixels: usize,
    element_count: usize,
}

fn pathops_candidate(
    outline: &Outline,
    scale: f32,
    mode: PathOpsMode,
    fill_type: PathFillType,
) -> Option<Outline> {
    let scaled = scale_outline(outline, scale);
    let (path, _) = super::build_skia_path(&scaled, false, fill_type)?;
    let simplified = match mode {
        PathOpsMode::Union => path.op(&path, PathOp::Union),
        PathOpsMode::Simplify => path.simplify(),
        PathOpsMode::FoldUnion => fold_union(&scaled),
    }
    .or_else(|| path.simplify())?;
    let simplified = simplified.as_winding().unwrap_or(simplified);
    let scaled_outline = outline_from_path_scaled(&simplified, 1.0);
    let oriented_outline = winding_from_even_odd(&scaled_outline);
    Some(scale_outline(&oriented_outline, 1.0 / scale))
}

fn fold_union(outline: &Outline) -> Option<Path> {
    let mut iter = outline.subpaths().iter();
    let first = subpath_to_skia_path(iter.next()?);
    Some(iter.fold(first, |acc, subpath| {
        let path = subpath_to_skia_path(subpath);
        acc.op(&path, PathOp::Union).unwrap_or(acc)
    }))
}

fn score_candidate(original_winding: &[u8], candidate: &Outline) -> Option<CandidateScore> {
    let candidate_winding = render_bitmap(
        candidate,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    )
    .ok()?;
    let candidate_even_odd = render_bitmap(
        candidate,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::EvenOdd,
    )
    .ok()?;
    Some(CandidateScore {
        overlap_pixels: hard_diff(&candidate_winding.data, &candidate_even_odd.data),
        mismatch_pixels: hard_diff(original_winding, &candidate_winding.data),
        element_count: candidate
            .subpaths()
            .iter()
            .map(|subpath| subpath.elements().len())
            .sum(),
    })
}

fn hard_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(a, b)| (**a == 255 && **b == 0) || (**a == 0 && **b == 255))
        .count()
}

fn scale_outline(outline: &Outline, scale: f32) -> Outline {
    let subpaths = outline
        .subpaths()
        .iter()
        .map(|s| {
            let start = Point::new(s.start().x * scale, s.start().y * scale);
            let elements = s
                .elements()
                .iter()
                .map(|e| match *e {
                    PathElement::LineTo(p) => {
                        PathElement::LineTo(Point::new(p.x * scale, p.y * scale))
                    }
                    PathElement::QuadTo { control, end } => PathElement::QuadTo {
                        control: Point::new(control.x * scale, control.y * scale),
                        end: Point::new(end.x * scale, end.y * scale),
                    },
                    PathElement::CurveTo {
                        control0,
                        control1,
                        end,
                    } => PathElement::CurveTo {
                        control0: Point::new(control0.x * scale, control0.y * scale),
                        control1: Point::new(control1.x * scale, control1.y * scale),
                        end: Point::new(end.x * scale, end.y * scale),
                    },
                })
                .collect();
            Subpath::new(start, elements, s.is_closed())
        })
        .collect();
    Outline::new(subpaths)
}

fn outline_from_path_scaled(path: &Path, scale: f32) -> Outline {
    let mut subpaths = Vec::new();
    let mut start = None;
    let mut elements = Vec::new();
    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => {
                if let Some(start) = start.replace(point_scaled(points[0], scale)) {
                    subpaths.push(Subpath::new(start, std::mem::take(&mut elements), false));
                }
            }
            PathVerb::Line => elements.push(PathElement::LineTo(point_scaled(points[1], scale))),
            PathVerb::Quad => elements.push(PathElement::QuadTo {
                control: point_scaled(points[1], scale),
                end: point_scaled(points[2], scale),
            }),
            PathVerb::Cubic => elements.push(PathElement::CurveTo {
                control0: point_scaled(points[1], scale),
                control1: point_scaled(points[2], scale),
                end: point_scaled(points[3], scale),
            }),
            PathVerb::Close => {
                if let Some(start) = start.take() {
                    subpaths.push(Subpath::new(start, std::mem::take(&mut elements), true));
                }
            }
            PathVerb::Conic => unreachable!("PathOps should not emit conic segments"),
        }
    }
    if let Some(start) = start {
        subpaths.push(Subpath::new(start, elements, false));
    }
    Outline::new(subpaths)
}

fn needs_pathops(outline: &Outline) -> bool {
    let subpaths = outline.subpaths();
    let n = subpaths.len();
    if n == 0 {
        return false;
    }
    if n == 1 {
        return true;
    }
    if n > 20 {
        return true;
    }

    let areas: Vec<f64> = subpaths.iter().map(subpath_area).collect();
    let skia_paths: Vec<Path> = subpaths.iter().map(subpath_to_skia_path).collect();
    let bounds: Vec<Rect> = skia_paths.iter().map(|path| *path.bounds()).collect();
    let interior_points: Vec<Option<SkPoint>> = skia_paths
        .iter()
        .zip(&bounds)
        .map(|(path, bounds)| interior_point(path, *bounds))
        .collect();

    if skia_paths.iter().any(subpath_simplifies) {
        return true;
    }

    let mut nesting = vec![0usize; n];
    for j in 0..n {
        for i in 0..n {
            if i == j {
                continue;
            }
            if subpath_contains(&skia_paths[i], bounds[i], bounds[j], interior_points[j]) {
                nesting[j] += 1;
            }
        }
    }

    let orientation_key = areas.iter().zip(&nesting).find_map(|(area, depth)| {
        (area.abs() > f64::EPSILON).then_some((*area < 0.0) ^ ((depth & 1) == 1))
    });
    if let Some(key) = orientation_key {
        for (area, depth) in areas.iter().zip(&nesting) {
            if area.abs() > f64::EPSILON && ((*area < 0.0) ^ ((depth & 1) == 1)) != key {
                return true;
            }
        }
    }

    for i in 0..n {
        for j in (i + 1)..n {
            if !bounds_intersect(bounds[i], bounds[j]) {
                continue;
            }
            let i_contains_j =
                subpath_contains(&skia_paths[i], bounds[i], bounds[j], interior_points[j]);
            let j_contains_i =
                subpath_contains(&skia_paths[j], bounds[j], bounds[i], interior_points[i]);
            if i_contains_j || j_contains_i {
                continue;
            }
            if bounds_contains(bounds[i], bounds[j]) || bounds_contains(bounds[j], bounds[i]) {
                return true;
            }
            return true;
        }
    }

    false
}

fn subpath_simplifies(path: &Path) -> bool {
    if path.is_convex() {
        return false;
    }
    let Some(simplified) = path.op(path, PathOp::Union).or_else(|| path.simplify()) else {
        return false;
    };
    path_signature(path) != path_signature(&simplified)
}

fn path_signature(path: &Path) -> Vec<PathVerb> {
    path.iter().map(|record| record.verb()).collect()
}

fn winding_from_even_odd(outline: &Outline) -> Outline {
    let subpaths = outline.subpaths();
    let n = subpaths.len();
    if n == 0 {
        return Outline::default();
    }

    let areas: Vec<f64> = subpaths.iter().map(subpath_area).collect();
    let skia_paths: Vec<Path> = subpaths.iter().map(subpath_to_skia_path).collect();
    let bounds: Vec<Rect> = skia_paths.iter().map(|path| *path.bounds()).collect();
    let interior_points: Vec<Option<SkPoint>> = skia_paths
        .iter()
        .zip(&bounds)
        .map(|(path, bounds)| interior_point(path, *bounds))
        .collect();

    let mut nesting = vec![0usize; n];
    for j in 0..n {
        for i in 0..n {
            if i == j {
                continue;
            }
            if subpath_contains(&skia_paths[i], bounds[i], bounds[j], interior_points[j]) {
                nesting[j] += 1;
            }
        }
    }

    let result: Vec<Subpath> = (0..n)
        .map(|i| {
            let subpath = &subpaths[i];
            let is_clockwise = areas[i] < 0.0;
            let is_even = (nesting[i] & 1) == 0;
            let should_reverse = true ^ is_clockwise ^ is_even;
            if should_reverse {
                reverse_subpath(subpath)
            } else {
                subpath.clone()
            }
        })
        .collect();

    Outline::new(result)
}

fn subpath_area(subpath: &Subpath) -> f64 {
    let start = subpath.start();
    let mut p0 = start;
    let mut value = 0.0f64;
    for element in subpath.elements() {
        match *element {
            PathElement::LineTo(p1) => {
                value -= ((p1.x - p0.x) * (p1.y + p0.y) * 0.5) as f64;
                p0 = p1;
            }
            PathElement::QuadTo { control, end } => {
                let (x0, y0) = (p0.x, p0.y);
                let (x1, y1) = (control.x - x0, control.y - y0);
                let (x2, y2) = (end.x - x0, end.y - y0);
                value -= ((x2 * y1 - x1 * y2) / 3.0) as f64;
                value -= ((end.x - x0) * (end.y + y0) * 0.5) as f64;
                p0 = end;
            }
            PathElement::CurveTo {
                control0,
                control1,
                end,
            } => {
                let (x0, y0) = (p0.x, p0.y);
                let (x1, y1) = (control0.x - x0, control0.y - y0);
                let (x2, y2) = (control1.x - x0, control1.y - y0);
                let (x3, y3) = (end.x - x0, end.y - y0);
                value -=
                    ((x1 * (-y2 - y3) + x2 * (y1 - 2.0 * y3) + x3 * (y1 + 2.0 * y2)) * 0.15) as f64;
                value -= ((end.x - x0) * (end.y + y0) * 0.5) as f64;
                p0 = end;
            }
        }
    }
    value -= ((start.x - p0.x) * (start.y + p0.y) * 0.5) as f64;
    value
}

fn subpath_to_skia_path(subpath: &Subpath) -> Path {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::EvenOdd);
    let start = subpath.start();
    builder.move_to((start.x, start.y));
    for element in subpath.elements() {
        match *element {
            PathElement::LineTo(point) => builder.line_to((point.x, point.y)),
            PathElement::QuadTo { control, end } => {
                builder.quad_to((control.x, control.y), (end.x, end.y))
            }
            PathElement::CurveTo {
                control0,
                control1,
                end,
            } => builder.cubic_to(
                (control0.x, control0.y),
                (control1.x, control1.y),
                (end.x, end.y),
            ),
        };
    }
    if subpath.is_closed() {
        builder.close();
    }
    builder.detach()
}

fn subpath_contains(
    outer: &Path,
    outer_bounds: Rect,
    inner_bounds: Rect,
    inner_point: Option<SkPoint>,
) -> bool {
    bounds_contains(outer_bounds, inner_bounds)
        && inner_point.is_some_and(|point| outer.contains(point))
}

fn bounds_contains(outer: Rect, inner: Rect) -> bool {
    !inner.is_empty()
        && !outer.is_empty()
        && outer.left <= inner.left
        && outer.top <= inner.top
        && outer.right >= inner.right
        && outer.bottom >= inner.bottom
}

fn bounds_intersect(a: Rect, b: Rect) -> bool {
    !a.is_empty()
        && !b.is_empty()
        && a.left <= b.right
        && b.left <= a.right
        && a.top <= b.bottom
        && b.top <= a.bottom
}

fn interior_point(path: &Path, bounds: Rect) -> Option<SkPoint> {
    if bounds.is_empty() {
        return None;
    }

    let center = SkPoint::new(
        (bounds.left + bounds.right) * 0.5,
        (bounds.top + bounds.bottom) * 0.5,
    );
    if path.contains(center) {
        return Some(center);
    }

    let width = bounds.right - bounds.left;
    let height = bounds.bottom - bounds.top;
    for grid in [3.0, 5.0, 9.0] {
        for y in 1..grid as i32 {
            for x in 1..grid as i32 {
                let point = SkPoint::new(
                    bounds.left + width * (x as f32 / grid),
                    bounds.top + height * (y as f32 / grid),
                );
                if path.contains(point) {
                    return Some(point);
                }
            }
        }
    }

    None
}

fn point_scaled(point: SkPoint, scale: f32) -> Point {
    Point::new(point.x * scale, point.y * scale)
}
