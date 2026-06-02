use std::collections::HashMap;

use i_overlay::core::fill_rule::FillRule;
use i_overlay::core::overlay_rule::OverlayRule;
use i_overlay::float::single::SingleFloatOverlay;
use skia_safe::{PathFillType, PathVerb};

use crate::geom::{Outline, PathElement, Point, Subpath};
use crate::skia::render_bitmap::render_fixed_path;

const BITMAP_SIZE: u32 = 128;
const FLATTEN_TOL: f32 = 0.002;
// Scale sweep (failures at 100 K): 16384->219, 131072->180, 262144->199, 524288->220.
// 131072 is the sweet spot.
const PATHOPS_SCALE: f32 = 131_072.0;

pub(crate) fn remove_overlaps(outline: &Outline) -> Outline {
    // Render winding; skip empty outlines.
    let Some(mut path) = build_path(outline, PathFillType::Winding) else {
        return outline.clone();
    };
    let winding = render_fixed_path(&mut path, BITMAP_SIZE, PathFillType::Winding);
    if !winding.contains(&255) {
        return outline.clone();
    }
    // Detect overlaps: winding != even-odd -> |w| >= 2 at some pixel.
    // Misses rare |w| >= 3 odd; the score check below catches those if they survive.
    // NOTE: a bbox-disjoint pre-check was tried but caused 0.7% failures at 100 K: subpath
    // bboxes can be mutually disjoint while one subpath self-intersects (e.g. figure-8 crossbar),
    // which the bbox check misses but the even-odd render catches.
    // NOTE: segment-level cp-bbox check (O(N²)) was tried but failed for adjacent bezier crossings
    // (e.g. Alexandria "five"): adjacent segments are skipped but CAN cross, causing has_overlaps.
    // NOTE: lower-res overlap prechecks lost accuracy or speed: 64px -> 223 failures at 100 K;
    // 96px stayed under threshold once but was slower due to the extra 128px original render.
    let even_odd = render_fixed_path(&mut path, BITMAP_SIZE, PathFillType::EvenOdd);
    if !even_odd
        .iter()
        .zip(&winding)
        .any(|(a, b)| (*a == 255) != (*b == 255))
    {
        return outline.clone();
    }

    // Stage 1: Skia PathOps simplify (preserves curves; fails on sub-pixel geometry).
    // Primary scale is 131072 (empirical sweet spot). The old ov-gated 65536 retry was
    // removed after two 100 K runs stayed under threshold; extra PathOps did not buy speed.
    let mut overlap_free: Option<Outline> = None;
    match pathops_simplify(&mut path, PATHOPS_SCALE) {
        None => {}
        Some(result) => {
            let (mm, ov) = candidate_stats(&winding, &result);
            if mm == 0 {
                return result;
            }
            if ov == 0 {
                overlap_free = Some(result);
            }
        }
    }
    // NOTE: removing this ov-gated alt-scale/overlap-free path and always falling through to
    // polygon_union was fast but failed badly: 34.5% has_overlaps at 100 K.
    // NOTE: skipping overlap_count and keeping the first mismatched PathOps result was not faster
    // and regressed to 930/100096 failures (0.9291%), mostly complex CJK has_overlaps.

    // Stage 2: Polygon union — overlap-free by construction; restores curves where endpoints match.
    // Older progressive coarser tolerances were unnecessary in current 100 K runs; the first
    // successful union is the only one used.
    if let Some(result) = polygon_union(outline, FLATTEN_TOL) {
        let mm = mismatch(&winding, &result);
        if mm == 0 {
            return result;
        }
        if overlap_free.is_none() {
            overlap_free = Some(result);
        }
    }

    overlap_free.unwrap_or_else(|| outline.clone())
}

// --- Bitmap helpers -----------------------------------------------------------

fn render(outline: &Outline, fill: PathFillType) -> Option<Vec<u8>> {
    let mut path = build_path(outline, fill)?;
    Some(render_fixed_path(&mut path, BITMAP_SIZE, fill))
}

fn build_path(outline: &Outline, fill: PathFillType) -> Option<skia_safe::Path> {
    super::build_skia_path(outline, false, fill).map(|(path, _)| path)
}

fn count_diff(a: &[u8], b: &[u8]) -> usize {
    a.iter()
        .zip(b)
        .filter(|(x, y)| (**x == 255) != (**y == 255))
        .count()
}

// Returns (bitmap mismatch against the original winding render, overlap pixels).
// Overlap: w!=0 (winding=255) AND |w| even (even-odd=0) => |w|>=2 even overlap.
// Misses |w|>=3 odd — same limitation as the pre-check; verified empirically not to occur.
fn candidate_stats(original_winding: &[u8], candidate: &Outline) -> (usize, usize) {
    let Some(mut path) = build_path(candidate, PathFillType::Winding) else {
        return (usize::MAX, usize::MAX);
    };
    let winding = render_fixed_path(&mut path, BITMAP_SIZE, PathFillType::Winding);
    let mismatch = count_diff(&winding, original_winding);
    if mismatch == 0 {
        return (0, 0);
    }
    let even_odd = render_fixed_path(&mut path, BITMAP_SIZE, PathFillType::EvenOdd);
    let overlaps = winding
        .iter()
        .zip(&even_odd)
        .filter(|&(&w, &e)| w == 255 && e == 0)
        .count();
    (mismatch, overlaps)
}

// Mismatch only; polygon union guarantees ov = 0 by construction.
fn mismatch(original_winding: &[u8], candidate: &Outline) -> usize {
    render(candidate, PathFillType::Winding)
        .map(|bmp| count_diff(&bmp, original_winding))
        .unwrap_or(usize::MAX)
}

// --- Skia PathOps ------------------------------------------------------------

fn pathops_simplify(path: &mut skia_safe::Path, scale: f32) -> Option<Outline> {
    path.set_fill_type(PathFillType::Winding);
    let path = path.try_make_scale((scale, scale))?;
    let result = path.simplify()?;
    // as_winding fixes contour orientations so output uses winding semantics.
    let oriented = result.as_winding().unwrap_or(result);
    let unscaled = oriented.try_make_scale((scale.recip(), scale.recip()))?;
    skia_path_to_outline(&unscaled)
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

// --- Polygon union fallback --------------------------------------------------
// Flattens curves to line segments at `tol`, unions via i_overlay (NonZero fill),
// then restores original curve elements where both endpoints exactly match the originals.
//
// Notes from earlier experiments:
//   Normalising all contours to CCW before union: 514 failures (fills holes).
//   1000x coord scale for i_overlay: marginal improvement (203->202, within noise).
//   10000x coord scale for i_overlay: regression (2.0->2.3/run in 20-run batches).
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

// Vertex map: float-bits key -> (subpath index, element index; 0=start, k>=1=elements[k-1].end)
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
    let clip: Vec<Vec<[f64; 2]>> = Vec::new();
    let shapes = contours.overlay(&clip, OverlayRule::Union, FillRule::NonZero);
    (!shapes.is_empty()).then(|| orient_shapes(shapes))
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
    // Forward: A->B is element vi_a.
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

// --- Curve flattening --------------------------------------------------------

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
