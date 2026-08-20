use skia_safe::{Path, PathBuilder, PathFillType, PathVerb};

use super::subpath::reverse_subpath;
use crate::outline::{BezPath, Bounds, PathEl, Point, bounds_from_subpath, subpath_is_closed};
use kurbo::Shape;

// TorchFont outlines are normalized to roughly em-sized coordinates. PathOps is
// more reliable at conventional font-unit magnitudes, so simplify a scaled copy.
const PATHOPS_SCALE: f32 = 131_072.0;

pub(crate) fn remove_overlaps(outline: &BezPath) -> BezPath {
    simplify(outline).unwrap_or_else(|| outline.clone())
}

pub(crate) fn random_remove_overlaps(outline: &BezPath, random_values: &[f32]) -> BezPath {
    let source: Vec<_> = outline.subpaths().collect();
    let bounds: Vec<_> = source
        .iter()
        .map(|subpath| bounds_from_subpath(subpath))
        .collect();
    let mut parent: Vec<_> = (0..source.len()).collect();

    for left in 0..source.len() {
        if !subpath_is_closed(source[left]) {
            continue;
        }
        for right in left + 1..source.len() {
            if subpath_is_closed(source[right]) && bounds_overlap(bounds[left], bounds[right]) {
                union(&mut parent, left, right);
            }
        }
    }
    for index in 0..parent.len() {
        parent[index] = find(&mut parent, index);
    }

    let mut members = vec![Vec::new(); source.len()];
    for (index, &root) in parent.iter().enumerate() {
        members[root].push(index);
    }
    let groups: Vec<Vec<_>> = members
        .into_iter()
        .filter(|group: &Vec<_>| group.len() > 1)
        .collect();

    if groups.is_empty() {
        return outline.clone();
    }

    let values = &random_values[..groups.len()];
    let mut selected: Vec<_> = groups
        .iter()
        .enumerate()
        .map(|(index, _)| values[index] < 0.5)
        .collect();
    if !selected.iter().any(|&value| value) {
        let index = values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(0, |(index, _)| index);
        selected[index] = true;
    }

    let mut group_of = vec![None; source.len()];
    for (group_index, group) in groups.iter().enumerate() {
        for &index in group {
            group_of[index] = Some(group_index);
        }
    }

    let mut result = BezPath::new();
    for index in 0..source.len() {
        let Some(group_index) = group_of[index] else {
            result.extend(source[index].iter().copied());
            continue;
        };
        let group = &groups[group_index];
        if group[0] != index {
            continue;
        }
        if selected[group_index] {
            let mut component = BezPath::new();
            for &other in group {
                component.extend(source[other].iter().copied());
            }
            let simplified = simplify(&component).unwrap_or(component);
            result.extend(simplified.elements().iter().copied());
        } else {
            for &other in group {
                result.extend(source[other].iter().copied());
            }
        }
    }
    result
}

fn bounds_overlap(a: Bounds, b: Bounds) -> bool {
    a.x_min < b.x_max && b.x_min < a.x_max && a.y_min < b.y_max && b.y_min < a.y_max
}

fn find(parent: &mut [usize], index: usize) -> usize {
    if parent[index] != index {
        parent[index] = find(parent, parent[index]);
    }
    parent[index]
}

fn union(parent: &mut [usize], left: usize, right: usize) {
    let left = find(parent, left);
    let right = find(parent, right);
    if left != right {
        parent[right] = left.min(right);
        parent[left] = left.min(right);
    }
}

fn simplify(outline: &BezPath) -> Option<BezPath> {
    let path = build_skia_path(outline)?;
    let scaled = path.try_make_scale((PATHOPS_SCALE, PATHOPS_SCALE))?;
    let simplified = scaled.simplify()?;

    // Simplify emits an even-odd path. Reorient nested contours before
    // returning to TorchFont, whose outlines use non-zero winding semantics.
    let simplified = outline_from_path(&simplified)?;
    let winding = winding_from_even_odd(&simplified);
    Some(kurbo::Affine::scale(f64::from(PATHOPS_SCALE.recip())) * &winding)
}

fn build_skia_path(outline: &BezPath) -> Option<Path> {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
    for &element in outline.elements() {
        push_skia_element(&mut builder, element);
    }
    (!builder.is_empty()).then(|| builder.detach())
}

// Explicitly close every contour, matching close_subpath's rationale: a
// contour is a fill boundary regardless of whether PathOps happened to emit
// a trailing Close. EvenOdd (rather than build_skia_path's Winding) is safe
// here only because every caller passes a single, simple, non-self-
// intersecting contour from Skia's own simplify() output; both fill rules
// agree for such a shape.
fn subpath_skia_path(subpath: &[PathEl]) -> Option<Path> {
    let mut builder = PathBuilder::new_with_fill_type(PathFillType::EvenOdd);
    for &element in subpath {
        push_skia_element(&mut builder, element);
    }
    builder.close();
    (!builder.is_empty()).then(|| builder.detach())
}

fn push_skia_element(builder: &mut PathBuilder, element: PathEl) {
    match element {
        PathEl::MoveTo(point) => {
            builder.move_to((point.x as f32, point.y as f32));
        }
        PathEl::LineTo(point) => {
            builder.line_to((point.x as f32, point.y as f32));
        }
        PathEl::QuadTo(control, end) => {
            builder.quad_to(
                (control.x as f32, control.y as f32),
                (end.x as f32, end.y as f32),
            );
        }
        PathEl::CurveTo(control0, control1, end) => {
            builder.cubic_to(
                (control0.x as f32, control0.y as f32),
                (control1.x as f32, control1.y as f32),
                (end.x as f32, end.y as f32),
            );
        }
        PathEl::ClosePath => {
            builder.close();
        }
    };
}

fn outline_from_path(path: &Path) -> Option<BezPath> {
    let mut outline = BezPath::new();
    let mut start = None;
    let mut elements: Vec<PathEl> = Vec::new();

    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => {
                commit_subpath(&mut outline, &mut start, &mut elements, false);
                start = Some(point(points[0]));
            }
            PathVerb::Line => elements.push(PathEl::LineTo(point(points[1]))),
            PathVerb::Quad => elements.push(PathEl::QuadTo(point(points[1]), point(points[2]))),
            PathVerb::Cubic => elements.push(PathEl::CurveTo(
                point(points[1]),
                point(points[2]),
                point(points[3]),
            )),
            PathVerb::Close => commit_subpath(&mut outline, &mut start, &mut elements, true),
            PathVerb::Conic => return None,
        }
    }
    commit_subpath(&mut outline, &mut start, &mut elements, false);

    (!outline.elements().is_empty()).then_some(outline)
}

fn commit_subpath(
    outline: &mut BezPath,
    start: &mut Option<Point>,
    elements: &mut Vec<PathEl>,
    closed: bool,
) {
    if let Some(start) = start.take()
        && !elements.is_empty()
    {
        outline.move_to(start);
        outline.extend(std::mem::take(elements));
        if closed {
            outline.close_path();
        }
    }
}

fn point(point: skia_safe::Point) -> Point {
    Point::new(point.x.into(), point.y.into())
}

fn winding_from_even_odd(outline: &BezPath) -> BezPath {
    // skia-pathops uses nesting depth instead of Skia's AsWinding operation;
    // keep that behavior while using TorchFont's native outline types.
    // Contours are treated as implicitly closed for area purposes regardless
    // of whether PathOps happened to emit a trailing Close: kurbo only
    // synthesizes the closing edge when ClosePath is present, but an open
    // polyline isn't a meaningful fill boundary.
    let mut contours: Vec<_> = outline
        .subpaths()
        .map(|subpath| (close_subpath(subpath).area(), subpath))
        .collect();
    contours.sort_by(|a, b| {
        b.0.abs()
            .partial_cmp(&a.0.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let bounding_boxes: Vec<_> = contours
        .iter()
        .map(|(_, subpath)| subpath.bounding_box())
        .collect();
    // The nesting loop below queries, for every candidate outer/inner pair,
    // whether the inner contour's points fall inside the outer one. Skia's
    // Path::contains is markedly faster per query than kurbo's analytic
    // per-segment winding, so build each contour's Skia path once up front
    // rather than converting it on every query.
    let skia_paths: Vec<_> = contours
        .iter()
        .map(|(_, subpath)| subpath_skia_path(subpath))
        .collect();

    let mut nesting = vec![0usize; contours.len()];
    for inner in 0..contours.len() {
        for outer in 0..inner {
            let is_inside = bounding_boxes[outer].contains_rect(bounding_boxes[inner])
                && skia_paths[outer]
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

fn close_subpath(subpath: &[PathEl]) -> BezPath {
    let mut path = BezPath::from_vec(subpath.to_vec());
    if !subpath_is_closed(subpath) {
        path.close_path();
    }
    path
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
    use kurbo::{BezPath, Rect, Shape};
    use skia_safe::{PathBuilder, PathFillType};

    use super::{outline_from_path, path_is_inside};

    fn rectangle(rect: Rect) -> BezPath {
        rect.to_path(0.1)
    }

    #[test]
    fn close_subpath_treats_open_subpath_as_implicitly_closed_for_area() {
        // Offset from the origin: kurbo's raw (unclosed) area only matches
        // the true polygon area when the missing closing edge happens to
        // pass through the origin, so this triangle is chosen to actually
        // exercise the implicit-closure behavior rather than mask it.
        let mut open = BezPath::new();
        open.move_to((1.0, 1.0));
        open.line_to((5.0, 1.0));
        open.line_to((1.0, 5.0));

        let raw_area = open.elements().area();
        let closed_area = super::close_subpath(open.elements()).area();

        assert_ne!(raw_area, closed_area);
        assert!((closed_area.abs() - 8.0).abs() < 1e-9);
    }

    #[test]
    fn subpath_skia_path_fills_open_subpath_interior() {
        // No close_path(): the edge back to (0.0, 0.0) is only implicit in
        // the input, but subpath_skia_path closes every contour explicitly.
        let mut outer = BezPath::new();
        outer.move_to((0.0, 0.0));
        outer.line_to((10.0, 0.0));
        outer.line_to((10.0, 10.0));
        outer.line_to((0.0, 10.0));

        let path = super::subpath_skia_path(outer.elements()).expect("subpath has segments");

        assert!(path.contains((5.0, 5.0)));
    }

    #[test]
    fn drops_degenerate_move_only_subpath() {
        let mut builder = PathBuilder::new_with_fill_type(PathFillType::Winding);
        builder.move_to((5.0, 5.0));
        builder.move_to((0.0, 0.0));
        builder.line_to((10.0, 0.0));
        builder.line_to((10.0, 10.0));
        builder.close();
        let path = builder.detach();

        let outline = outline_from_path(&path).expect("path has real segments");
        let subpath_count = outline.subpaths().count();

        assert_eq!(subpath_count, 1);
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
}
