use kurbo::Shape;
use skia_safe::{Path, PathFillType, PathOp};

use super::skia::{PATHOPS_SCALE, build_skia_path_builder};
use crate::outline::{
    BezPath, Bounds, PathEl, Point, bounds_from_subpath, subpath_is_closed, subpath_start,
};

pub(crate) fn reverse_subpath(subpath: &[PathEl]) -> BezPath {
    BezPath::from_vec(subpath.to_vec()).reverse_subpaths()
}

pub(crate) fn reverse_closed_subpaths(outline: &BezPath) -> BezPath {
    let mut result = BezPath::new();
    for subpath in outline.subpaths() {
        if subpath_is_closed(subpath) {
            let reversed = reverse_subpath(subpath);
            result.extend(reversed.elements().iter().copied());
        } else {
            result.extend(subpath.iter().copied());
        }
    }
    result
}

pub(crate) fn normalize_winding(
    outline: &BezPath,
    clockwise: bool,
) -> Result<BezPath, &'static str> {
    // Grouping is unnecessary when every group must make the same decision:
    // all areas already match, or all contours are closed and strictly opposite.
    let mut preserve = true;
    let mut reverse = true;
    for path in outline.subpaths() {
        let area = subpath_area(path);
        let aligned = area == 0.0 || (area < 0.0) == clockwise;
        preserve &= aligned;
        reverse &= !aligned && subpath_is_closed(path);
        if !preserve && !reverse {
            break;
        }
    }
    if preserve {
        return Ok(outline.clone());
    }
    if reverse {
        return Ok(reverse_closed_subpaths(outline));
    }
    transform_groups(outline, |_, group, paths| {
        let mut largest_area = 0.0_f64;
        for &index in group {
            let area = subpath_area(paths[index]);
            if area.abs() > largest_area.abs() {
                largest_area = area;
            }
        }
        Ok(largest_area != 0.0 && (largest_area < 0.0) != clockwise)
    })
}

pub(crate) fn reverse_winding_groups(
    outline: &BezPath,
    reversal_mask: &[bool],
) -> Result<BezPath, &'static str> {
    // There can be at most one group per subpath. A uniform mask covering
    // that upper bound makes every group's choice known without intersections.
    let count = outline.subpaths().count();
    if let Some(mask) = reversal_mask.get(..count) {
        if mask.iter().all(|&selected| !selected) {
            return Ok(outline.clone());
        }
        if mask.iter().all(|&selected| selected) && outline.subpaths().all(subpath_is_closed) {
            return Ok(reverse_closed_subpaths(outline));
        }
    }
    transform_groups(outline, |index, _, _| {
        reversal_mask
            .get(index)
            .copied()
            .ok_or("reversal_mask must contain at least one value per winding group")
    })
}

fn transform_groups(
    outline: &BezPath,
    choose_reverse: impl Fn(usize, &[usize], &[&[PathEl]]) -> Result<bool, &'static str>,
) -> Result<BezPath, &'static str> {
    let paths: Vec<_> = outline.subpaths().collect();
    let groups = winding_groups(&paths)?;
    let mut reverse = vec![false; paths.len()];
    for (index, group) in groups.into_iter().enumerate() {
        let selected = choose_reverse(index, &group, &paths)?;
        if selected && group.iter().all(|&index| subpath_is_closed(paths[index])) {
            for index in group {
                reverse[index] = true;
            }
        }
    }
    let mut result = BezPath::new();
    for (path, reverse) in paths.into_iter().zip(reverse) {
        if reverse {
            result.extend(reverse_subpath(path).elements().iter().copied());
        } else {
            result.extend(path.iter().copied());
        }
    }
    Ok(result)
}

fn winding_groups(paths: &[&[PathEl]]) -> Result<Vec<Vec<usize>>, &'static str> {
    if paths.len() <= 1 {
        return Ok((0..paths.len()).map(|index| vec![index]).collect());
    }
    // Convert candidate contours once; segment bounds can rule out
    // boundary crossings before the more expensive PathOps intersection.
    let bounds: Vec<_> = paths
        .iter()
        .map(|path| rect_from_bounds(bounds_from_subpath(path)))
        .collect();
    let mut skia_paths = vec![None; paths.len()];
    let mut boundary_boxes: Vec<Option<Vec<kurbo::Rect>>> = vec![None; paths.len()];
    let mut boundary_segments: Vec<Option<Vec<kurbo::PathSeg>>> = vec![None; paths.len()];
    let mut nonempty = vec![None; paths.len()];
    let mut roots: Vec<_> = (0..paths.len()).collect();
    let mut order: Vec<_> = (0..paths.len()).collect();
    order.sort_unstable_by(|&a, &b| bounds[a].x0.total_cmp(&bounds[b].x0).then(a.cmp(&b)));
    for (position, &left) in order.iter().enumerate() {
        let a = bounds[left];
        for &right in &order[position + 1..] {
            let b = bounds[right];
            // Later contours start at least as far right. Once there is no
            // positive-width overlap, every remaining pair is disjoint too.
            if b.x0 >= a.x1 {
                break;
            }
            if !a.overlaps(b) || root(&mut roots, left) == root(&mut roots, right) {
                continue;
            }
            let overlap = a.intersect(b);
            if overlap.area() == 0.0 {
                continue;
            }
            let mut intersects = None;
            for (outer, inner) in [(left, right), (right, left)] {
                let boundary =
                    boundary_boxes[outer].get_or_insert_with(|| boundary_bounds(paths[outer]));
                let b = bounds[inner];
                // No boundary can enter this rectangle, so the outer winding
                // is constant throughout it. A single containment query suffices.
                if boundary.iter().all(|rect| !rect.overlaps(b)) {
                    if skia_paths[outer].is_none() {
                        skia_paths[outer] = Some(grouping_path(paths[outer])?);
                    }
                    let point = subpath_start(paths[inner]);
                    let inside = skia_paths[outer].as_ref().unwrap().contains((
                        point.x as f32 * PATHOPS_SCALE,
                        point.y as f32 * PATHOPS_SCALE,
                    ));
                    if !inside {
                        intersects = Some(false);
                        break;
                    }
                    if let Some(nonempty) = *nonempty[inner].get_or_insert_with(|| {
                        subpath_has_fill(paths[inner], &mut skia_paths[inner])
                    }) {
                        intersects = Some(nonempty);
                        break;
                    }
                }
            }
            if intersects.is_none() {
                for index in [left, right] {
                    if skia_paths[index].is_none() {
                        skia_paths[index] = Some(grouping_path(paths[index])?);
                    }
                }
            }
            if intersects.is_none() {
                let center = overlap.center();
                let x = center.x as f32;
                let y = center.y as f32;
                let point = kurbo::Point::new(f64::from(x), f64::from(y));
                let probe = kurbo::Rect::from_points(point, point);
                // Boundary exclusion proves a neighborhood of the point belongs to
                // both fills: this is a sufficient proof, not approximate sampling.
                // Exclude all boundary boxes so shared edges cannot qualify.
                if [left, right].iter().all(|&index| {
                    boundary_boxes[index]
                        .as_ref()
                        .unwrap()
                        .iter()
                        .all(|rect| !rect.overlaps(probe))
                        && skia_paths[index]
                            .as_ref()
                            .unwrap()
                            .contains((x * PATHOPS_SCALE, y * PATHOPS_SCALE))
                }) {
                    intersects = Some(true);
                }
            }
            if intersects.is_none() {
                // Disjoint segment bounds prove the boundaries cannot meet,
                // even when neither contour fits in the other's bounding box.
                for index in [left, right] {
                    boundary_segments[index].get_or_insert_with(|| {
                        kurbo::segments(
                            paths[index]
                                .iter()
                                .copied()
                                .chain(std::iter::once(PathEl::ClosePath)),
                        )
                        .collect()
                    });
                }
                let boundaries_disjoint = boundary_boxes[left]
                    .as_ref()
                    .unwrap()
                    .iter()
                    .zip(boundary_segments[left].as_ref().unwrap())
                    .all(|(a, segment_a)| {
                        boundary_boxes[right]
                            .as_ref()
                            .unwrap()
                            .iter()
                            .zip(boundary_segments[right].as_ref().unwrap())
                            .all(|(b, segment_b)| {
                                !a.overlaps(*b) || control_hulls_disjoint(*segment_a, *segment_b)
                            })
                    });
                if boundaries_disjoint {
                    let mut filled_overlap = false;
                    let mut determined = true;
                    // Either contour may enclose the other. Query both sides;
                    // an outside point alone cannot rule out containment.
                    for (outer, inner) in [(left, right), (right, left)] {
                        let point = subpath_start(paths[inner]);
                        if skia_paths[outer].as_ref().unwrap().contains((
                            point.x as f32 * PATHOPS_SCALE,
                            point.y as f32 * PATHOPS_SCALE,
                        )) {
                            match *nonempty[inner].get_or_insert_with(|| {
                                subpath_has_fill(paths[inner], &mut skia_paths[inner])
                            }) {
                                Some(true) => {
                                    filled_overlap = true;
                                    break;
                                }
                                Some(false) => {}
                                None => determined = false,
                            }
                        }
                    }
                    if determined || filled_overlap {
                        intersects = Some(filled_overlap);
                    }
                }
            }
            let intersects = if let Some(intersects) = intersects {
                intersects
            } else {
                let intersection = skia_paths[left]
                    .as_ref()
                    .unwrap()
                    .op(skia_paths[right].as_ref().unwrap(), PathOp::Intersect)
                    .ok_or("could not determine winding groups")?;
                !intersection.is_empty()
            };
            if intersects {
                let left = root(&mut roots, left);
                let right = root(&mut roots, right);
                roots[left.max(right)] = left.min(right);
            }
        }
    }
    let mut members = vec![Vec::new(); paths.len()];
    for index in 0..paths.len() {
        let group = root(&mut roots, index);
        members[group].push(index);
    }
    Ok(members
        .into_iter()
        .filter(|group| !group.is_empty())
        .collect())
}

// A Bezier curve lies in the convex hull of its control points. A strict
// separating line between those hulls proves the curves cannot meet. Robust
// orientation predicates keep near-collinear edges and touching hulls safe.
fn control_hulls_disjoint(a: kurbo::PathSeg, b: kurbo::PathSeg) -> bool {
    fn points(segment: kurbo::PathSeg) -> ([Point; 4], usize) {
        match segment {
            kurbo::PathSeg::Line(line) => ([line.p0, line.p1, Point::ZERO, Point::ZERO], 2),
            kurbo::PathSeg::Quad(quad) => ([quad.p0, quad.p1, quad.p2, Point::ZERO], 3),
            kurbo::PathSeg::Cubic(cubic) => ([cubic.p0, cubic.p1, cubic.p2, cubic.p3], 4),
        }
    }
    fn separates(a: &[Point], b: &[Point]) -> bool {
        for (i, p) in a.iter().enumerate() {
            for q in &a[i + 1..] {
                let side = |r: &Point| {
                    robust::orient2d(
                        robust::Coord { x: p.x, y: p.y },
                        robust::Coord { x: q.x, y: q.y },
                        robust::Coord { x: r.x, y: r.y },
                    )
                };
                let sides = a
                    .iter()
                    .map(side)
                    .fold((true, true), |(positive, negative), s| {
                        (positive && s >= 0.0, negative && s <= 0.0)
                    });
                if (sides.0 && b.iter().all(|r| side(r) < 0.0))
                    || (sides.1 && b.iter().all(|r| side(r) > 0.0))
                {
                    return true;
                }
            }
        }
        false
    }
    let (a, na) = points(a);
    let (b, nb) = points(b);
    separates(&a[..na], &b[..nb]) || separates(&b[..nb], &a[..na])
}

fn subpath_has_fill(path: &[PathEl], skia_path: &mut Option<Path>) -> Option<bool> {
    if skia_path.is_none() {
        *skia_path = Some(grouping_path(path).ok()?);
    }
    let skia_path = skia_path.as_ref().unwrap();
    let center = skia_path.bounds().center();
    let point = Point::new(
        f64::from(center.x) / f64::from(PATHOPS_SCALE),
        f64::from(center.y) / f64::from(PATHOPS_SCALE),
    );
    let probe = kurbo::PathSeg::Line(kurbo::Line::new(point, point));
    // An inside point separated from every boundary has a filled neighborhood.
    // Signed area is insufficient here: cancellation in retraced curves can
    // leave a nonzero floating-point residual even when their fill is empty.
    if skia_path.contains(center)
        && kurbo::segments(
            path.iter()
                .copied()
                .chain(std::iter::once(PathEl::ClosePath)),
        )
        .all(|segment| control_hulls_disjoint(segment, probe))
    {
        return Some(true);
    }
    skia_path.simplify().map(|path| !path.is_empty())
}

fn grouping_path(path: &[PathEl]) -> Result<Path, &'static str> {
    build_skia_path_builder(path, PathFillType::Winding)
        .detach()
        .try_make_scale((PATHOPS_SCALE, PATHOPS_SCALE))
        .ok_or("could not scale contours for winding grouping")
}

// Kurbo computes curve extrema; retain outward-rounded f32 bounds for Skia.
fn boundary_bounds(path: &[PathEl]) -> Vec<kurbo::Rect> {
    kurbo::segments(
        path.iter()
            .copied()
            .chain(std::iter::once(PathEl::ClosePath)),
    )
    .map(|segment| {
        let rect = segment.bounding_box();
        let mut bounds = Bounds::new(Point::new(rect.x0, rect.y0));
        bounds.include(Point::new(rect.x1, rect.y1));
        rect_from_bounds(bounds)
    })
    .collect()
}

fn rect_from_bounds(bounds: Bounds) -> kurbo::Rect {
    kurbo::Rect::new(
        bounds.x_min.into(),
        bounds.y_min.into(),
        bounds.x_max.into(),
        bounds.y_max.into(),
    )
}

fn root(roots: &mut [usize], mut index: usize) -> usize {
    while roots[index] != index {
        roots[index] = roots[roots[index]];
        index = roots[index];
    }
    index
}

// Explicitly close every contour, matching subpath_area's rationale: a
// contour is a fill boundary regardless of whether PathOps happened to emit
// a trailing Close. EvenOdd (rather than build_skia_path's Winding) is safe
// here only because every caller passes a single, simple, non-self-
// intersecting contour from Skia's own simplify() output; both fill rules
// agree for such a shape.
fn subpath_skia_path(subpath: &[PathEl]) -> Option<Path> {
    let mut builder = build_skia_path_builder(subpath, PathFillType::EvenOdd);
    builder.close();
    (!builder.is_empty()).then(|| builder.detach())
}

pub(crate) fn winding_from_even_odd(outline: &BezPath) -> BezPath {
    // Contours are treated as implicitly closed for area purposes regardless
    // of whether PathOps happened to emit a trailing Close: kurbo only
    // synthesizes the closing edge when ClosePath is present, but an open
    // polyline isn't a meaningful fill boundary.
    let mut contours: Vec<_> = outline
        .subpaths()
        .map(|subpath| (subpath_area(subpath), subpath))
        .collect();
    if let [(area, subpath)] = contours.as_slice() {
        return if *area < 0.0 {
            reverse_subpath(subpath)
        } else {
            outline.clone()
        };
    }
    contours.sort_by(|a, b| {
        b.0.abs()
            .partial_cmp(&a.0.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bounding_boxes: Vec<_> = contours
        .iter()
        .map(|(_, subpath)| subpath.bounding_box())
        .collect();
    // Most contours have no children. Build a containment path only when
    // another contour's tight bounds actually fit inside it.
    let mut skia_paths = vec![None; contours.len()];

    let mut nesting = vec![0usize; contours.len()];
    for inner in 0..contours.len() {
        for outer in 0..inner {
            let is_inside = bounding_boxes[outer].contains_rect(bounding_boxes[inner])
                && skia_paths[outer]
                    .get_or_insert_with(|| subpath_skia_path(contours[outer].1))
                    .as_ref()
                    .is_some_and(|path| contour_is_inside(path, contours[inner].1));
            if is_inside {
                nesting[inner] += 1;
            }
        }
    }

    let mut result = BezPath::new();
    for ((area, subpath), depth) in contours.into_iter().zip(nesting) {
        let is_clockwise = area < 0.0;
        let is_outer = depth.is_multiple_of(2);
        if is_clockwise == is_outer {
            result.extend(reverse_subpath(subpath).elements().iter().copied());
        } else {
            result.extend(subpath.iter().copied());
        }
    }
    result
}

fn subpath_area(subpath: &[PathEl]) -> f64 {
    if subpath_is_closed(subpath) {
        return subpath.area();
    }
    kurbo::segments(
        subpath
            .iter()
            .copied()
            .chain(std::iter::once(PathEl::ClosePath)),
    )
    .map(|segment| segment.area())
    .sum()
}

#[cfg(test)]
fn path_is_inside(outer: &[PathEl], inner: &[PathEl]) -> bool {
    // PathOps simplification has already split intersecting contours, so a
    // contour whose tight bounds fit and whose on-curve points are inside is
    // nested. Checking tight-bound containment also catches curves that bulge
    // outside the candidate parent while keeping their endpoints inside.
    outer.bounding_box().contains_rect(inner.bounding_box())
        && subpath_skia_path(outer).is_some_and(|path| contour_is_inside(&path, inner))
}

fn contour_is_inside(outer: &Path, inner: &[PathEl]) -> bool {
    inner
        .iter()
        .filter_map(kurbo::PathEl::end_point)
        .all(|point| outer.contains((point.x as f32, point.y as f32)))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::{
        Point, outline_from_subpaths, subpath_elements, subpath_from_elements, subpath_start,
    };
    use kurbo::Rect;

    fn pt(x: f32, y: f32) -> Point {
        Point::new(x.into(), y.into())
    }

    fn line(x: f32, y: f32) -> PathEl {
        PathEl::LineTo(pt(x, y))
    }

    fn closed(start: Point, elements: Vec<PathEl>) -> BezPath {
        subpath_from_elements(start, elements, true)
    }

    fn open(start: Point, elements: Vec<PathEl>) -> BezPath {
        subpath_from_elements(start, elements, false)
    }

    fn subpaths(path: &BezPath) -> Vec<BezPath> {
        path.subpaths()
            .map(|subpath| BezPath::from_vec(subpath.to_vec()))
            .collect()
    }

    fn rectangle(rect: Rect) -> BezPath {
        rect.to_path(0.1)
    }

    #[test]
    fn control_hulls_separate_oblique_lines_and_bezier_curves() {
        let diagonal = kurbo::PathSeg::Line(kurbo::Line::new((0.0, 0.0), (4.0, 4.0)));
        let curves = [
            kurbo::PathSeg::Line(kurbo::Line::new((0.0, 1.0), (3.0, 4.0))),
            kurbo::PathSeg::Quad(kurbo::QuadBez::new((0.0, 1.0), (1.0, 4.0), (3.0, 4.0))),
            kurbo::PathSeg::Cubic(kurbo::CubicBez::new(
                (0.0, 1.0),
                (1.0, 4.0),
                (2.0, 4.0),
                (3.0, 4.0),
            )),
        ];
        for curve in curves {
            assert!(diagonal.bounding_box().overlaps(curve.bounding_box()));
            assert!(control_hulls_disjoint(diagonal, curve));
            assert!(control_hulls_disjoint(curve, diagonal));
        }
    }

    #[test]
    fn control_hulls_keep_crossings_tangencies_and_retraced_segments() {
        let horizontal = kurbo::PathSeg::Line(kurbo::Line::new((0.0, 0.0), (4.0, 0.0)));
        for curve in [
            kurbo::PathSeg::Line(kurbo::Line::new((2.0, -1.0), (2.0, 1.0))),
            kurbo::PathSeg::Line(kurbo::Line::new((3.0, 0.0), (1.0, 0.0))),
            kurbo::PathSeg::Quad(kurbo::QuadBez::new((1.0, 1.0), (2.0, -1.0), (3.0, 1.0))),
            kurbo::PathSeg::Cubic(kurbo::CubicBez::new(
                (1.0, 1.0),
                (2.0, -1.0),
                (3.0, -1.0),
                (4.0, 1.0),
            )),
        ] {
            assert!(!control_hulls_disjoint(horizontal, curve));
            assert!(!control_hulls_disjoint(curve, horizontal));
        }
    }

    #[test]
    fn control_hulls_distinguish_touching_from_one_ulp_gaps() {
        let a = kurbo::PathSeg::Line(kurbo::Line::new((0.0, 1.0), (1.0, 1.0)));
        let gap = f64::from(1.0_f32.next_up());
        let b = kurbo::PathSeg::Line(kurbo::Line::new((0.0, gap), (1.0, gap)));
        assert!(control_hulls_disjoint(a, b));
        assert!(!control_hulls_disjoint(a, a));
    }

    #[test]
    fn grouping_matches_pathops_for_containment_crossings_and_empty_fills() {
        let contours = [
            rectangle(Rect::new(0.0, 0.0, 10.0, 10.0)),
            rectangle(Rect::new(2.0, 2.0, 8.0, 8.0)),
            rectangle(Rect::new(10.0, 0.0, 12.0, 10.0)),
            rectangle(Rect::new(-1.0, 4.0, 11.0, 6.0)),
            closed(
                pt(2.0, 2.0),
                vec![line(8.0, 8.0), line(2.0, 8.0), line(8.0, 2.0)],
            ),
            closed(
                pt(2.0, 2.0),
                vec![line(8.0, 2.0), line(2.0, 8.0), line(8.0, 2.0)],
            ),
            closed(
                pt(2.0, 5.0),
                vec![PathEl::CurveTo(pt(2.0, 20.0), pt(8.0, 20.0), pt(8.0, 5.0))],
            ),
            closed(
                pt(0.0, 0.0),
                vec![line(4.0, 4.0), line(4.0, 2.0), line(0.0, -2.0)],
            ),
            closed(
                pt(0.0, 1.0),
                vec![line(4.0, 5.0), line(4.0, 4.5), line(0.0, 0.5)],
            ),
            closed(
                pt(0.5, 0.0),
                vec![line(3.5, 3.0), line(3.5, 2.5), line(0.5, -0.5)],
            ),
        ];
        for left in &contours {
            for right in &contours {
                let paths = [left.elements(), right.elements()];
                let skia: Vec<_> = paths
                    .iter()
                    .map(|path| {
                        build_skia_path_builder(path, PathFillType::Winding)
                            .detach()
                            .try_make_scale((PATHOPS_SCALE, PATHOPS_SCALE))
                            .unwrap()
                    })
                    .collect();
                let intersects = !skia[0].op(&skia[1], PathOp::Intersect).unwrap().is_empty();
                assert_eq!(
                    winding_groups(&paths).unwrap().len(),
                    if intersects { 1 } else { 2 },
                    "left={left:?}, right={right:?}"
                );
            }
        }
    }

    #[test]
    fn reverse_subpath_empty_returns_clone() {
        let s = open(pt(0.0, 0.0), vec![]);
        let r = reverse_subpath(s.elements());
        assert_eq!(r, s);
    }

    #[test]
    fn reverse_subpath_triangle() {
        let s = open(pt(0.0, 0.0), vec![line(1.0, 0.0), line(0.5, 1.0)]);
        let r = reverse_subpath(s.elements());
        assert_eq!(subpath_start(r.elements()), pt(0.5, 1.0));
        assert_eq!(subpath_elements(r.elements())[0], line(1.0, 0.0));
        assert_eq!(subpath_elements(r.elements())[1], line(0.0, 0.0));
    }

    #[test]
    fn reverse_subpath_cubic_swaps_controls() {
        let s = open(
            pt(0.0, 0.0),
            vec![PathEl::CurveTo(pt(1.0, 2.0), pt(3.0, 4.0), pt(5.0, 0.0))],
        );
        let r = reverse_subpath(s.elements());
        assert_eq!(subpath_start(r.elements()), pt(5.0, 0.0));
        assert_eq!(
            subpath_elements(r.elements())[0],
            PathEl::CurveTo(pt(3.0, 4.0), pt(1.0, 2.0), pt(0.0, 0.0))
        );
    }

    #[test]
    fn reverse_closed_subpaths_skips_open() {
        let subpath = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let outline = outline_from_subpaths([subpath.clone()]);
        let result = reverse_closed_subpaths(&outline);
        assert_eq!(subpaths(&result)[0], subpath);
    }

    #[test]
    fn reverse_closed_subpaths_reverses_closed() {
        let outline =
            outline_from_subpaths([closed(pt(0.0, 0.0), vec![line(1.0, 0.0), line(0.5, 1.0)])]);
        let result = reverse_closed_subpaths(&outline);
        let result_subpaths = subpaths(&result);
        let s = &result_subpaths[0];
        assert_eq!(subpath_start(s.elements()), pt(0.5, 1.0));
        assert!(subpath_is_closed(s.elements()));
    }

    #[test]
    fn subpath_area_treats_open_subpath_as_implicitly_closed() {
        // Offset from the origin: kurbo's raw (unclosed) area only matches
        // the true polygon area when the missing closing edge happens to
        // pass through the origin, so this triangle is chosen to actually
        // exercise the implicit-closure behavior rather than mask it.
        let mut open = BezPath::new();
        open.move_to((1.0, 1.0));
        open.line_to((5.0, 1.0));
        open.line_to((1.0, 5.0));

        let raw_area = open.elements().area();
        let closed_area = super::subpath_area(open.elements());

        assert_ne!(raw_area, closed_area);
        assert!((closed_area.abs() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn subpath_skia_path_fills_open_subpath_interior() {
        let mut outer = BezPath::new();
        outer.move_to((0.0, 0.0));
        outer.line_to((10.0, 0.0));
        outer.line_to((10.0, 10.0));
        outer.line_to((0.0, 10.0));

        let path = super::subpath_skia_path(outer.elements()).expect("subpath has segments");

        assert!(path.contains((5.0, 5.0)));
    }

    #[test]
    fn recognizes_contained_path() {
        let outer = rectangle(Rect::new(0.0, 0.0, 10.0, 10.0));
        let inner = rectangle(Rect::new(2.0, 2.0, 8.0, 8.0));

        assert!(path_is_inside(outer.elements(), inner.elements()));
    }

    #[test]
    fn rejects_path_with_only_start_inside() {
        let outer = rectangle(Rect::new(0.0, 0.0, 10.0, 10.0));
        let mut crossing = BezPath::new();
        crossing.move_to((5.0, 5.0));
        crossing.line_to((12.0, 5.0));
        crossing.line_to((12.0, 8.0));
        crossing.close_path();

        assert!(!path_is_inside(outer.elements(), crossing.elements()));
    }

    #[test]
    fn rejects_curve_with_endpoints_inside_but_body_outside() {
        let outer = rectangle(Rect::new(0.0, 0.0, 10.0, 10.0));
        let mut crossing = BezPath::new();
        crossing.move_to((2.0, 5.0));
        crossing.curve_to((2.0, 20.0), (8.0, 20.0), (8.0, 5.0));
        crossing.close_path();

        assert!(!path_is_inside(outer.elements(), crossing.elements()));
    }

    #[test]
    fn winding_handles_nested_contours_and_disjoint_siblings() {
        let mut outline = BezPath::new();
        for rect in [
            Rect::new(2.0, 2.0, 4.0, 4.0),
            Rect::new(12.0, 0.0, 14.0, 2.0),
            Rect::new(0.0, 0.0, 10.0, 10.0),
            Rect::new(6.0, 6.0, 9.0, 9.0),
            Rect::new(1.0, 1.0, 5.0, 5.0),
        ] {
            outline.extend(rectangle(rect).elements().iter().copied());
        }

        let winding = super::winding_from_even_odd(&outline);

        // Two holes share the outer contour; one hole contains an island.
        for (point, expected) in [
            ((0.5, 0.5), 1),
            ((1.5, 1.5), 0),
            ((3.0, 3.0), 1),
            ((7.0, 7.0), 0),
            ((13.0, 1.0), 1),
            ((11.0, 1.0), 0),
        ] {
            assert_eq!(winding.winding(point.into()), expected);
        }
    }
}
