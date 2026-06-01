use std::collections::HashMap;

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use skia_safe::{PathFillType, PathVerb};

use crate::geom::{Outline, PathElement, Point, Subpath};
use crate::skia::render_bitmap::{RenderMode, render_bitmap};

const BITMAP_SIZE: u32 = 128;
const FLATTEN_TOL: f32 = 0.002;
// Scale sweep (failures at 100 K): 16384→219, 131072→180, 262144→199, 524288→220.
// 131072 is the sweet spot; dual-scale retry regressed to 206.
const PATHOPS_SCALE: f32 = 131_072.0;

// Outer rect that covers the entire FIXED render window [-0.25, 1.25]² with large margin.
// Prepending CW shifts every pixel's winding w → w-1; CCW shifts w → w+1.
// With the original winding render, three-way AND detects |w| ≥ 2 (matches test logic):
//   original==255 (w≠0) & with_cw==255 (w≠1) & with_ccw==255 (w≠-1)  →  |w| ≥ 2.
const OUTER: [[f32; 2]; 4] = [[-4.0, -4.0], [-4.0, 3.5], [13.0, 3.5], [13.0, -4.0]];

pub(crate) fn remove_overlaps(outline: &Outline) -> Outline {
    // Render winding; skip empty outlines.
    let Some(winding) = render(outline, PathFillType::Winding) else {
        return outline.clone();
    };
    if !winding.contains(&255) {
        return outline.clone();
    }
    // Detect overlaps: winding ≠ even-odd ↔ even |w| ≥ 2.
    // Odd |w| ≥ 3 overlaps are rare in fonts; even-|w| detection covers the common case.
    let has_overlap =
        render(outline, PathFillType::EvenOdd).is_some_and(|eo| any_differ(&winding, &eo));
    if !has_overlap {
        return outline.clone();
    }

    // Stage 1: Skia PathOps simplify (preserves curves; fails on sub-pixel geometry).
    let mut best: Option<(usize, usize, Outline)> = None;
    if let Some(result) = pathops_simplify(outline, PATHOPS_SCALE) {
        let (ov, mm) = score(&winding, &result);
        if ov == 0 && mm == 0 {
            return result;
        }
        best = Some((ov, mm, result));
    }

    // Stage 2: Polygon union — guaranteed overlap-free by construction; restore curves after.
    // Try progressively coarser tolerances to handle near-degenerate geometry (e.g. complex CJK).
    for &tol in &[
        FLATTEN_TOL,
        FLATTEN_TOL * 5.0,
        FLATTEN_TOL * 25.0,
        FLATTEN_TOL * 125.0,
    ] {
        let Some(result) = polygon_union(outline, tol) else {
            continue;
        };
        let mm = mismatch(&winding, &result);
        if mm == 0 {
            return result;
        }
        if best.as_ref().is_none_or(|(bo, bm, _)| (0, mm) < (*bo, *bm)) {
            best = Some((0, mm, result));
        }
        break; // Use first successful union; coarser tols only tried when union fails entirely.
    }

    best.map(|(_, _, r)| r).unwrap_or_else(|| outline.clone())
}

// ─── Bitmap helpers ───────────────────────────────────────────────────────────

fn render(outline: &Outline, fill: PathFillType) -> Option<Vec<u8>> {
    render_bitmap(outline, BITMAP_SIZE, RenderMode::Fixed, fill)
        .ok()
        .map(|b| b.data)
}

fn any_differ(a: &[u8], b: &[u8]) -> bool {
    a.iter().zip(b).any(|(x, y)| (*x == 255) != (*y == 255))
}

fn count_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (**x == 255) != (**y == 255))
        .count()
}

/// Score a candidate outline against the original winding bitmap.
/// Returns (overlap_px, mismatch_px).
/// Uses the same 3-render outer-rect method as the test to catch all |w| ≥ 2 overlaps
/// (winding-vs-even-odd only catches even |w|; odd |w| ≥ 3 can slip through).
fn score(original_winding: &[u8], candidate: &Outline) -> (usize, usize) {
    let Some(cw) = render(candidate, PathFillType::Winding) else {
        return (usize::MAX, usize::MAX);
    };
    let mismatch = count_diff(&cw, original_winding);

    // Overlap check: prepend CW then CCW outer rect; overlap = pixel filled in all three.
    let cw_rect = with_outer(candidate, true);
    let ccw_rect = with_outer(candidate, false);
    let Some(cw_render) = render(&cw_rect, PathFillType::Winding) else {
        return (usize::MAX, usize::MAX);
    };
    let Some(ccw_render) = render(&ccw_rect, PathFillType::Winding) else {
        return (usize::MAX, usize::MAX);
    };
    let overlap = cw
        .iter()
        .zip(&cw_render)
        .zip(&ccw_render)
        .filter(|((a, b), c)| **a == 255 && **b == 255 && **c == 255)
        .count();

    (overlap, mismatch)
}

/// Mismatch only; caller guarantees overlap = 0 (polygon union by construction).
fn mismatch(original_winding: &[u8], candidate: &Outline) -> usize {
    render(candidate, PathFillType::Winding)
        .map(|bmp| count_diff(&bmp, original_winding))
        .unwrap_or(usize::MAX)
}

/// Prepend a large CW (clockwise, y-up) or CCW outer rectangle to the outline.
fn with_outer(outline: &Outline, clockwise: bool) -> Outline {
    let [a, b, c, d] = OUTER.map(|[x, y]| Point::new(x, y));
    let (v0, v1, v2, v3) = if clockwise {
        (a, b, c, d)
    } else {
        (a, d, c, b)
    };
    let rect = Subpath::new(
        v0,
        vec![
            PathElement::LineTo(v1),
            PathElement::LineTo(v2),
            PathElement::LineTo(v3),
        ],
        true,
    );
    let mut subpaths = vec![rect];
    subpaths.extend_from_slice(outline.subpaths());
    Outline::new(subpaths)
}

// ─── Skia PathOps ────────────────────────────────────────────────────────────

fn pathops_simplify(outline: &Outline, scale: f32) -> Option<Outline> {
    let scaled = scale_outline(outline, scale);
    let (path, _) = super::build_skia_path(&scaled, false, PathFillType::Winding)?;
    let result = path.simplify()?;
    // as_winding fixes contour orientations so output uses winding semantics.
    let oriented = result.as_winding().unwrap_or(result);
    let unscaled = skia_path_to_outline(&oriented)?;
    Some(scale_outline(&unscaled, 1.0 / scale))
}

fn scale_outline(outline: &Outline, s: f32) -> Outline {
    Outline::new(
        outline
            .subpaths()
            .iter()
            .map(|sp| {
                Subpath::new(
                    spt(sp.start(), s),
                    sp.elements().iter().map(|&e| scale_elem(e, s)).collect(),
                    sp.is_closed(),
                )
            })
            .collect(),
    )
}

#[inline]
fn spt(p: Point, s: f32) -> Point {
    Point::new(p.x * s, p.y * s)
}

fn scale_elem(e: PathElement, s: f32) -> PathElement {
    match e {
        PathElement::LineTo(p) => PathElement::LineTo(spt(p, s)),
        PathElement::QuadTo { control, end } => PathElement::QuadTo {
            control: spt(control, s),
            end: spt(end, s),
        },
        PathElement::CurveTo {
            control0,
            control1,
            end,
        } => PathElement::CurveTo {
            control0: spt(control0, s),
            control1: spt(control1, s),
            end: spt(end, s),
        },
    }
}

fn skia_path_to_outline(path: &skia_safe::Path) -> Option<Outline> {
    let mut subpaths: Vec<Subpath> = Vec::new();
    let mut cur_start: Option<skia_safe::Point> = None;
    let mut cur_elems: Vec<PathElement> = Vec::new();

    for rec in path.iter() {
        let pts = rec.points();
        match rec.verb() {
            PathVerb::Move => {
                commit(&mut cur_start, &mut cur_elems, &mut subpaths, false);
                cur_start = Some(pts[0]);
            }
            PathVerb::Line => cur_elems.push(PathElement::LineTo(skp(pts[1]))),
            PathVerb::Quad => cur_elems.push(PathElement::QuadTo {
                control: skp(pts[1]),
                end: skp(pts[2]),
            }),
            PathVerb::Cubic => cur_elems.push(PathElement::CurveTo {
                control0: skp(pts[1]),
                control1: skp(pts[2]),
                end: skp(pts[3]),
            }),
            PathVerb::Close => commit(&mut cur_start, &mut cur_elems, &mut subpaths, true),
            PathVerb::Conic => return None,
        }
    }
    commit(&mut cur_start, &mut cur_elems, &mut subpaths, false);
    (!subpaths.is_empty()).then(|| Outline::new(subpaths))
}

fn commit(
    start: &mut Option<skia_safe::Point>,
    elems: &mut Vec<PathElement>,
    out: &mut Vec<Subpath>,
    closed: bool,
) {
    if let Some(s) = start.take()
        && !elems.is_empty()
    {
        out.push(Subpath::new(skp(s), std::mem::take(elems), closed));
    }
}

#[inline]
fn skp(pt: skia_safe::Point) -> Point {
    Point::new(pt.x, pt.y)
}

// ─── Polygon union fallback ───────────────────────────────────────────────────
// Flattens curves to line segments at `tol`, unions via i_overlay (NonZero fill),
// then restores original curve elements where both endpoints exactly match the originals.
//
// Notes from earlier experiments:
//   Normalising all contours to CCW before union: 514 failures (fills holes).
//   1000× coord scale for i_overlay: marginal improvement (203→202, within noise).
//   PathOps on line-only polygon: same Skia sub-pixel bug; spurious 128px overlaps.
//   Skipping degenerate contours in sequential fallback: bitmap_mismatch regression.

fn polygon_union(outline: &Outline, tol: f32) -> Option<Outline> {
    let (contours, vmap) = flatten_tagged(outline, tol);
    if contours.is_empty() {
        return None;
    }
    let oriented = union_contours(contours)?;
    let subpaths: Vec<Subpath> = oriented
        .iter()
        .filter(|c| c.len() >= 3)
        .map(|c| restore_curves(c, &vmap, outline))
        .collect();
    (!subpaths.is_empty()).then(|| Outline::new(subpaths))
}

// Vertex map: float-bits key → (subpath index, element index; 0=start, k≥1=elements[k-1].end)
type VertexMap = HashMap<(u32, u32), (usize, usize)>;

fn flatten_tagged(outline: &Outline, tol: f32) -> (Vec<Vec<[f64; 2]>>, VertexMap) {
    let mut contours = Vec::new();
    let mut vmap: VertexMap = HashMap::new();
    let tol2 = tol * tol;

    for (si, sp) in outline.subpaths().iter().enumerate() {
        let s = sp.start();
        let mut c = vec![[s.x as f64, s.y as f64]];
        vmap.entry((s.x.to_bits(), s.y.to_bits()))
            .or_insert((si, 0));
        let mut prev = [s.x, s.y];
        for (ei, &elem) in sp.elements().iter().enumerate() {
            let end = elem.end();
            match elem {
                PathElement::LineTo(_) => c.push([end.x as f64, end.y as f64]),
                PathElement::QuadTo { control: q, .. } => {
                    flatten_quad(prev, [q.x, q.y], [end.x, end.y], tol2, &mut c)
                }
                PathElement::CurveTo {
                    control0: c0,
                    control1: c1,
                    ..
                } => flatten_cubic(
                    prev,
                    [c0.x, c0.y],
                    [c1.x, c1.y],
                    [end.x, end.y],
                    tol2,
                    &mut c,
                ),
            }
            vmap.entry((end.x.to_bits(), end.y.to_bits()))
                .or_insert((si, ei + 1));
            prev = [end.x, end.y];
        }
        // Drop closing duplicate vertex.
        if c.len() > 1 {
            let last = *c.last().unwrap();
            if (last[0] - c[0][0]).abs() < 1e-7 && (last[1] - c[0][1]).abs() < 1e-7 {
                c.pop();
            }
        }
        if c.len() >= 3 {
            contours.push(c);
        }
    }
    (contours, vmap)
}

fn union_contours(contours: Vec<Vec<[f64; 2]>>) -> Option<Vec<Vec<[f64; 2]>>> {
    let clip: Vec<Vec<[f64; 2]>> = vec![];
    let shapes = contours
        .clone()
        .overlay(&clip, OverlayRule::Union, FillRule::NonZero);
    if !shapes.is_empty() {
        return Some(orient_shapes(shapes));
    }
    // Sequential fallback for near-degenerate polygons where bulk union returns empty.
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
                    if (signed_area(&c) > 0.0) == want_ccw {
                        c
                    } else {
                        c.into_iter().rev().collect()
                    }
                })
        })
        .collect()
}

fn signed_area(c: &[[f64; 2]]) -> f64 {
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

fn restore_curves(contour: &[[f64; 2]], vmap: &VertexMap, outline: &Outline) -> Subpath {
    let n = contour.len();
    let start = contour[0];
    let elems = (0..n)
        .map(|i| {
            let a = contour[i];
            let b = contour[(i + 1) % n];
            let ax = (a[0] as f32).to_bits();
            let ay = (a[1] as f32).to_bits();
            let bx = (b[0] as f32).to_bits();
            let by = (b[1] as f32).to_bits();
            try_restore(ax, ay, bx, by, vmap, outline)
                .unwrap_or_else(|| PathElement::LineTo(Point::new(b[0] as f32, b[1] as f32)))
        })
        .collect();
    Subpath::new(Point::new(start[0] as f32, start[1] as f32), elems, true)
}

fn try_restore(
    ax: u32,
    ay: u32,
    bx: u32,
    by: u32,
    vmap: &VertexMap,
    outline: &Outline,
) -> Option<PathElement> {
    let &(si_a, vi_a) = vmap.get(&(ax, ay))?;
    let &(si_b, vi_b) = vmap.get(&(bx, by))?;
    if si_a != si_b {
        return None;
    }
    let sp = &outline.subpaths()[si_a];
    let n = sp.elements().len();
    // Forward: A→B is element vi_a.
    if vi_b == vi_a + 1 || (vi_a + 1 == n && vi_b == 0) {
        return sp.elements().get(vi_a).copied();
    }
    // Backward: contour reversed by orient_shapes; element vi_b, reversed.
    let b = Point::new(f32::from_bits(bx), f32::from_bits(by));
    if vi_a == vi_b + 1 || (vi_b + 1 == n && vi_a == 0) {
        return sp.elements().get(vi_b).map(|e| e.reversed_to(b));
    }
    None
}

// ─── Curve flattening ─────────────────────────────────────────────────────────

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
