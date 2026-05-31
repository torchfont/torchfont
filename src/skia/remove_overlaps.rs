use crate::geom::reverse_subpath;
use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use skia_safe::{Path, PathFillType, PathOp, PathVerb};

use crate::geom::{Outline, PathElement, Point, Subpath};
use crate::skia::render_bitmap::{RenderMode, render_bitmap};

const PATHOPS_SCALE: f32 = 16384.0;
const SCORE_BITMAP_SIZE: u32 = 64;
const MAX_PATHOPS_ELEMENTS: usize = 2000;
const FLATTEN_TOLERANCE_LINETO: f32 = 0.002;
const OUTER_RECT_MIN: f32 = -2.0;
const OUTER_RECT_MAX: f32 = 3.0;

pub(crate) fn remove_overlaps(outline: &Outline) -> Outline {
    let Ok(orig_winding) = render_bitmap(
        outline,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    ) else {
        return outline.clone();
    };

    if !orig_winding.data.contains(&255) {
        return outline.clone();
    }

    // Two-path overlap detection:
    // • Same-sign multi-subpath: outer-rect catches |w|≥2 including w=3 (triple overlap).
    // • Otherwise (self-intersecting subpath): even-odd detects w=2 with one render.
    let has_overlap = if overlap_possible(outline) {
        outer_rect_overlap_count(&orig_winding.data, outline) > 0
    } else {
        render_bitmap(
            outline,
            SCORE_BITMAP_SIZE,
            RenderMode::Fixed,
            PathFillType::EvenOdd,
        )
        .map(|eo| hard_diff(&orig_winding.data, &eo.data) != 0)
        .unwrap_or(false)
    };
    if !has_overlap {
        return outline.clone();
    }

    let original = &orig_winding.data;
    let mut best: Option<(CandidateScore, Outline)> = None;
    let total_elements: usize = outline.subpaths().iter().map(|s| s.elements().len()).sum();

    // Stage 1: Skia PathOps — preserves bezier curves.
    if total_elements <= MAX_PATHOPS_ELEMENTS
        && let Some(r) = consider(
            pathops_candidate(outline, PATHOPS_SCALE, PathFillType::Winding),
            original,
            &mut best,
        )
    {
        return r;
    }

    // Stage 2: LineTo-only polygon. polygon_union enforces CCW outer / CW holes
    // at every step including the sequential fallback, so raw is usually correct.
    if is_imperfect(&best)
        && let Some(raw) = build_polygon_lineto(outline)
        && let Some(r) = consider(Some(raw), original, &mut best)
    {
        return r;
    }

    // Stage 3: Re-process the best visually-correct candidate.
    // When mismatch==0 but overlap>0, the shape is right but winding is wrong.
    if is_imperfect(&best)
        && let Some((score, ref candidate)) = best.clone()
        && score.mismatch_pixels == 0
    {
        if let Some(r) = consider(build_polygon_lineto(candidate), original, &mut best) {
            return r;
        }
        let nelems: usize = candidate
            .subpaths()
            .iter()
            .map(|s| s.elements().len())
            .sum();
        if nelems <= MAX_PATHOPS_ELEMENTS {
            // Two scales to cover Skia precision quirks on LineTo candidates.
            for scale in [PATHOPS_SCALE, 65536.0f32] {
                if let Some(r) = consider(
                    pathops_candidate(candidate, scale, PathFillType::Winding),
                    original,
                    &mut best,
                ) {
                    return r;
                }
            }
        }
    }

    best.map(|(_, c)| c).unwrap_or_else(|| outline.clone())
}

/// Score candidate; update best; return it when overlap=0 and mismatch=0.
fn consider(
    candidate: Option<Outline>,
    original: &[u8],
    best: &mut Option<(CandidateScore, Outline)>,
) -> Option<Outline> {
    let c = candidate?;
    let score = score_candidate(original, &c)?;
    if best.as_ref().is_none_or(|(bs, _)| score < *bs) {
        *best = Some((score, c.clone()));
    }
    (score.overlap_pixels == 0 && score.mismatch_pixels == 0).then_some(c)
}

fn is_imperfect(best: &Option<(CandidateScore, Outline)>) -> bool {
    best.as_ref()
        .is_none_or(|(s, _)| s.overlap_pixels != 0 || s.mismatch_pixels != 0)
}

fn hard_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter().zip(b).filter(|(a, b)| *a ^ *b == 255).count()
}

// ─── Overlap detection ───────────────────────────────────────────────────────

/// True when same-sign subpaths with overlapping control-point bboxes exist.
fn overlap_possible(outline: &Outline) -> bool {
    let sps = outline.subpaths();
    if sps.len() <= 1 {
        return false;
    }
    let areas: Vec<f64> = sps.iter().map(subpath_area).collect();
    let boxes: Vec<[f32; 4]> = sps.iter().map(subpath_control_bbox).collect();
    for i in 0..sps.len() {
        for j in (i + 1)..sps.len() {
            if areas[i] * areas[j] <= 0.0 {
                continue;
            }
            let [ax1, ay1, ax2, ay2] = boxes[i];
            let [bx1, by1, bx2, by2] = boxes[j];
            if ax1 <= bx2 && bx1 <= ax2 && ay1 <= by2 && by1 <= ay2 {
                return true;
            }
        }
    }
    false
}

fn subpath_control_bbox(sp: &Subpath) -> [f32; 4] {
    let mut xmin = sp.start().x;
    let mut ymin = sp.start().y;
    let mut xmax = xmin;
    let mut ymax = ymin;
    let mut expand = |p: Point| {
        xmin = xmin.min(p.x);
        ymin = ymin.min(p.y);
        xmax = xmax.max(p.x);
        ymax = ymax.max(p.y);
    };
    for &e in sp.elements() {
        match e {
            PathElement::LineTo(p) => expand(p),
            PathElement::QuadTo { control, end } => {
                expand(control);
                expand(end);
            }
            PathElement::CurveTo {
                control0,
                control1,
                end,
            } => {
                expand(control0);
                expand(control1);
                expand(end);
            }
        }
    }
    [xmin, ymin, xmax, ymax]
}

fn subpath_area(sp: &Subpath) -> f64 {
    let start = sp.start();
    let mut p0 = start;
    let mut v = 0.0f64;
    for &e in sp.elements() {
        let p1 = e.end();
        v -= ((p1.x - p0.x) * (p1.y + p0.y) * 0.5) as f64;
        match e {
            PathElement::QuadTo { control: c, .. } => {
                let (x1, y1) = ((c.x - p0.x) as f64, (c.y - p0.y) as f64);
                let (x2, y2) = ((p1.x - p0.x) as f64, (p1.y - p0.y) as f64);
                v -= (x2 * y1 - x1 * y2) / 3.0;
            }
            PathElement::CurveTo {
                control0: c0,
                control1: c1,
                ..
            } => {
                let (x1, y1) = ((c0.x - p0.x) as f64, (c0.y - p0.y) as f64);
                let (x2, y2) = ((c1.x - p0.x) as f64, (c1.y - p0.y) as f64);
                let (x3, y3) = ((p1.x - p0.x) as f64, (p1.y - p0.y) as f64);
                v -= (x1 * (-y2 - y3) + x2 * (y1 - 2.0 * y3) + x3 * (y1 + 2.0 * y2)) * 0.15;
            }
            _ => {}
        }
        p0 = p1;
    }
    v -= ((start.x - p0.x) * (start.y + p0.y) * 0.5) as f64;
    v
}

// ─── Candidate scoring ───────────────────────────────────────────────────────

#[derive(Clone, Copy, Eq, Ord, PartialEq, PartialOrd)]
struct CandidateScore {
    overlap_pixels: usize,
    mismatch_pixels: usize,
    element_count: usize,
}

fn score_candidate(original: &[u8], candidate: &Outline) -> Option<CandidateScore> {
    let winding = render_bitmap(
        candidate,
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    )
    .ok()?;
    let mismatch = hard_diff(original, &winding.data);
    // Processed candidates have |w|≤2 at most; even-odd detects w=2 with one render.
    let overlap = if mismatch == 0 {
        let evenodd = render_bitmap(
            candidate,
            SCORE_BITMAP_SIZE,
            RenderMode::Fixed,
            PathFillType::EvenOdd,
        )
        .ok()?;
        hard_diff(&winding.data, &evenodd.data)
    } else {
        0
    };
    Some(CandidateScore {
        overlap_pixels: overlap,
        mismatch_pixels: mismatch,
        element_count: candidate
            .subpaths()
            .iter()
            .map(|s| s.elements().len())
            .sum(),
    })
}

// ─── Overlap counting via outer-rect ─────────────────────────────────────────

/// Build an outline with a rectangular outer contour prepended.
/// CW in y-up decrements winding by 1; CCW in y-up increments it.
fn with_outer_rect(outline: &Outline, cw: bool) -> Outline {
    let (lo, hi) = (OUTER_RECT_MIN, OUTER_RECT_MAX);
    let corners: [(f32, f32); 3] = if cw {
        [(lo, hi), (hi, hi), (hi, lo)]
    } else {
        [(hi, lo), (hi, hi), (lo, hi)]
    };
    let rect = Subpath::new(
        Point::new(lo, lo),
        corners
            .iter()
            .map(|&(x, y)| PathElement::LineTo(Point::new(x, y)))
            .collect(),
        true,
    );
    Outline::new(
        std::iter::once(rect)
            .chain(outline.subpaths().iter().cloned())
            .collect(),
    )
}

/// Count pixels with |w|≥2: filled by winding AND by both outer-rect shifts.
fn outer_rect_overlap_count(winding_bmp: &[u8], outline: &Outline) -> usize {
    let cw = render_bitmap(
        &with_outer_rect(outline, true),
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    )
    .ok();
    let ccw = render_bitmap(
        &with_outer_rect(outline, false),
        SCORE_BITMAP_SIZE,
        RenderMode::Fixed,
        PathFillType::Winding,
    )
    .ok();
    let (Some(cw), Some(ccw)) = (cw, ccw) else {
        return 0;
    };
    winding_bmp
        .iter()
        .zip(&cw.data)
        .zip(&ccw.data)
        .filter(|((w, cw), ccw)| **w == 255 && **cw == 255 && **ccw == 255)
        .count()
}

// ─── Skia PathOps backend ────────────────────────────────────────────────────

fn pathops_candidate(outline: &Outline, scale: f32, fill_type: PathFillType) -> Option<Outline> {
    let scaled = map_outline(outline, |p| Point::new(p.x * scale, p.y * scale));
    let (path, _) = super::build_skia_path(&scaled, false, fill_type)?;
    let simplified = path.op(&path, PathOp::Union).or_else(|| path.simplify())?;
    let simplified = simplified.as_winding().unwrap_or(simplified);
    // Skia emits CCW in y-down (= CW in y-up); reverse to font convention.
    let result = reverse_all_subpaths(&skia_path_to_outline(&simplified, 1.0));
    Some(map_outline(&result, |p| {
        Point::new(p.x / scale, p.y / scale)
    }))
}

fn skia_path_to_outline(path: &Path, scale: f32) -> Outline {
    let mut subpaths = Vec::new();
    let mut start: Option<Point> = None;
    let mut elements = Vec::new();
    for record in path.iter() {
        let pts = record.points();
        match record.verb() {
            PathVerb::Move => {
                if let Some(s) = start.replace(Point::new(pts[0].x * scale, pts[0].y * scale)) {
                    subpaths.push(Subpath::new(s, std::mem::take(&mut elements), false));
                }
            }
            PathVerb::Line => elements.push(PathElement::LineTo(Point::new(
                pts[1].x * scale,
                pts[1].y * scale,
            ))),
            PathVerb::Quad => elements.push(PathElement::QuadTo {
                control: Point::new(pts[1].x * scale, pts[1].y * scale),
                end: Point::new(pts[2].x * scale, pts[2].y * scale),
            }),
            PathVerb::Cubic => elements.push(PathElement::CurveTo {
                control0: Point::new(pts[1].x * scale, pts[1].y * scale),
                control1: Point::new(pts[2].x * scale, pts[2].y * scale),
                end: Point::new(pts[3].x * scale, pts[3].y * scale),
            }),
            PathVerb::Close => {
                if let Some(s) = start.take() {
                    subpaths.push(Subpath::new(s, std::mem::take(&mut elements), true));
                }
            }
            PathVerb::Conic => unreachable!("PathOps should not emit conic segments"),
        }
    }
    if let Some(s) = start {
        subpaths.push(Subpath::new(s, elements, false));
    }
    Outline::new(subpaths)
}

// ─── Orientation helpers ─────────────────────────────────────────────────────

fn reverse_all_subpaths(outline: &Outline) -> Outline {
    Outline::new(outline.subpaths().iter().map(reverse_subpath).collect())
}

// ─── Geometry helper ─────────────────────────────────────────────────────────

fn map_outline(outline: &Outline, f: impl Fn(Point) -> Point) -> Outline {
    Outline::new(
        outline
            .subpaths()
            .iter()
            .map(|s| {
                let start = f(s.start());
                let elements = s
                    .elements()
                    .iter()
                    .map(|&e| match e {
                        PathElement::LineTo(p) => PathElement::LineTo(f(p)),
                        PathElement::QuadTo { control, end } => PathElement::QuadTo {
                            control: f(control),
                            end: f(end),
                        },
                        PathElement::CurveTo {
                            control0,
                            control1,
                            end,
                        } => PathElement::CurveTo {
                            control0: f(control0),
                            control1: f(control1),
                            end: f(end),
                        },
                    })
                    .collect();
                Subpath::new(start, elements, s.is_closed())
            })
            .collect(),
    )
}

// ─── Polygon backend ──────────────────────────────────────────────────────────

fn flatten_quad(p0: [f32; 2], p1: [f32; 2], p2: [f32; 2], tol2: f32, out: &mut Vec<[f64; 2]>) {
    let mid = [(p0[0] + p2[0]) * 0.5, (p0[1] + p2[1]) * 0.5];
    let (dx, dy) = (p1[0] - mid[0], p1[1] - mid[1]);
    if dx * dx + dy * dy <= tol2 {
        out.push([p2[0] as f64, p2[1] as f64]);
        return;
    }
    let q0 = [(p0[0] + p1[0]) * 0.5, (p0[1] + p1[1]) * 0.5];
    let q1 = [(p1[0] + p2[0]) * 0.5, (p1[1] + p2[1]) * 0.5];
    let r = [(q0[0] + q1[0]) * 0.5, (q0[1] + q1[1]) * 0.5];
    flatten_quad(p0, q0, r, tol2, out);
    flatten_quad(r, q1, p2, tol2, out);
}

fn flatten_cubic(
    p0: [f32; 2],
    c0: [f32; 2],
    c1: [f32; 2],
    p1: [f32; 2],
    tol2: f32,
    out: &mut Vec<[f64; 2]>,
) {
    let (dx, dy) = (p1[0] - p0[0], p1[1] - p0[1]);
    let len2 = dx * dx + dy * dy;
    let dc0 = (c0[0] - p0[0]) * dy - (c0[1] - p0[1]) * dx;
    let dc1 = (c1[0] - p0[0]) * dy - (c1[1] - p0[1]) * dx;
    if dc0 * dc0 <= tol2 * len2 * 9.0 && dc1 * dc1 <= tol2 * len2 * 9.0 {
        out.push([p1[0] as f64, p1[1] as f64]);
        return;
    }
    let q0 = [(p0[0] + c0[0]) * 0.5, (p0[1] + c0[1]) * 0.5];
    let q1 = [(c0[0] + c1[0]) * 0.5, (c0[1] + c1[1]) * 0.5];
    let q2 = [(c1[0] + p1[0]) * 0.5, (c1[1] + p1[1]) * 0.5];
    let r0 = [(q0[0] + q1[0]) * 0.5, (q0[1] + q1[1]) * 0.5];
    let r1 = [(q1[0] + q2[0]) * 0.5, (q1[1] + q2[1]) * 0.5];
    let s = [(r0[0] + r1[0]) * 0.5, (r0[1] + r1[1]) * 0.5];
    flatten_cubic(p0, q0, r0, s, tol2, out);
    flatten_cubic(s, r1, q2, p1, tol2, out);
}

fn flatten_outline(outline: &Outline, tol: f32) -> Vec<Vec<[f64; 2]>> {
    let mut all_contours = Vec::new();
    let tol2 = tol * tol;
    for sp in outline.subpaths() {
        let start = sp.start();
        let mut contour = vec![[start.x as f64, start.y as f64]];
        let mut prev = [start.x, start.y];
        for &elem in sp.elements() {
            let end = elem.end();
            match elem {
                PathElement::LineTo(_) => contour.push([end.x as f64, end.y as f64]),
                PathElement::QuadTo { control: c, end: e } => {
                    flatten_quad(prev, [c.x, c.y], [e.x, e.y], tol2, &mut contour);
                }
                PathElement::CurveTo {
                    control0: c0,
                    control1: c1,
                    end: e,
                } => {
                    flatten_cubic(
                        prev,
                        [c0.x, c0.y],
                        [c1.x, c1.y],
                        [e.x, e.y],
                        tol2,
                        &mut contour,
                    );
                }
            }
            prev = [end.x, end.y];
        }
        if contour.len() > 1 {
            let last = *contour.last().unwrap();
            if (last[0] - contour[0][0]).abs() < 1e-7 && (last[1] - contour[0][1]).abs() < 1e-7 {
                contour.pop();
            }
        }
        if contour.len() >= 3 {
            all_contours.push(contour);
        }
    }
    all_contours
}

// ─── Polygon union ────────────────────────────────────────────────────────────

/// Shoelace area (positive = CCW in y-up, negative = CW).
fn signed_area_f64(c: &[[f64; 2]]) -> f64 {
    let n = c.len();
    (0..n)
        .map(|i| {
            let [x0, y0] = c[i];
            let [x1, y1] = c[(i + 1) % n];
            x0 * y1 - x1 * y0
        })
        .sum::<f64>()
        * 0.5
}

/// Orient i_overlay shape output: outer (idx=0) → CCW, holes (idx>0) → CW.
fn orient_shapes(shapes: Vec<Vec<Vec<[f64; 2]>>>) -> Vec<Vec<[f64; 2]>> {
    shapes
        .into_iter()
        .flat_map(|shape| {
            shape
                .into_iter()
                .enumerate()
                .filter(|(_, c)| c.len() >= 3)
                .map(|(idx, c)| {
                    let want_ccw = idx == 0;
                    if (signed_area_f64(&c) > 0.0) == want_ccw {
                        c
                    } else {
                        c.into_iter().rev().collect()
                    }
                })
        })
        .collect()
}

/// Compute polygon union enforcing CCW outer / CW holes at every step.
fn polygon_union(contours: Vec<Vec<[f64; 2]>>) -> Option<Vec<Vec<[f64; 2]>>> {
    let clip: Vec<Vec<[f64; 2]>> = vec![];
    let shapes = contours
        .clone()
        .overlay(&clip, OverlayRule::Union, FillRule::NonZero);
    if !shapes.is_empty() {
        return Some(orient_shapes(shapes));
    }
    // Sequential fallback: enforce orientation at each step.
    let mut current = vec![contours[0].clone()];
    for next in contours.into_iter().skip(1) {
        let result = current.overlay(&[next], OverlayRule::Union, FillRule::NonZero);
        if result.is_empty() {
            return None;
        }
        current = orient_shapes(result);
    }
    Some(current)
}

fn build_polygon_lineto(outline: &Outline) -> Option<Outline> {
    // Try fine then coarser tolerance; coarser reduces near-collinear degeneracy for i_overlay.
    for &tol in &[FLATTEN_TOLERANCE_LINETO, FLATTEN_TOLERANCE_LINETO * 5.0] {
        let contours = flatten_outline(outline, tol);
        if contours.is_empty() {
            continue;
        }
        if let Some(shapes) = polygon_union(contours) {
            let subpaths: Vec<Subpath> = shapes
                .iter()
                .filter(|c| c.len() >= 3)
                .map(|c| {
                    let start = Point::new(c[0][0] as f32, c[0][1] as f32);
                    let elements = c[1..]
                        .iter()
                        .map(|v| PathElement::LineTo(Point::new(v[0] as f32, v[1] as f32)))
                        .collect();
                    Subpath::new(start, elements, true)
                })
                .collect();
            if !subpaths.is_empty() {
                return Some(Outline::new(subpaths));
            }
        }
    }
    None
}
