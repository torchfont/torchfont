use kurbo::Shape;
use skia_safe::{Path, PathFillType, PathOp};

use super::skia::{PATHOPS_SCALE, build_skia_path_builder};
use crate::outline::{BezPath, PathEl, bounds_from_subpath, subpath_is_closed};

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
    // Use kurbo's tight bounds before invoking Skia's intersection operations.
    // Convert each candidate contour at most once, and skip already joined pairs.
    let bounds: Vec<_> = paths.iter().map(|path| bounds_from_subpath(path)).collect();
    let mut skia_paths = vec![None; paths.len()];
    let mut roots: Vec<_> = (0..paths.len()).collect();
    for left in 0..paths.len() {
        for right in left + 1..paths.len() {
            let a = bounds[left];
            let b = bounds[right];
            if a.x_min >= b.x_max
                || b.x_min >= a.x_max
                || a.y_min >= b.y_max
                || b.y_min >= a.y_max
                || root(&mut roots, left) == root(&mut roots, right)
            {
                continue;
            }
            for index in [left, right] {
                if skia_paths[index].is_none() {
                    let mut builder = build_skia_path_builder(paths[index], PathFillType::Winding);
                    skia_paths[index] = Some(
                        builder
                            .detach()
                            .try_make_scale((PATHOPS_SCALE, PATHOPS_SCALE))
                            .ok_or("could not scale contours for winding grouping")?,
                    );
                }
            }
            let intersection = skia_paths[left]
                .as_ref()
                .unwrap()
                .op(skia_paths[right].as_ref().unwrap(), PathOp::Intersect)
                .ok_or("could not determine winding groups")?;
            if !intersection.is_empty() {
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

fn root(roots: &mut [usize], index: usize) -> usize {
    if roots[index] != index {
        roots[index] = root(roots, roots[index]);
    }
    roots[index]
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
