use skia_safe::{Path, PathFillType, PathVerb};

use super::render_bitmap::render_bitmap_in_bounds;
use super::skia::{PATHOPS_SCALE, build_skia_path};
use super::winding::winding_from_even_odd;
use crate::outline::{
    BezPath, Bounds, PathEl, Point, bounds_from_outline, bounds_from_subpath, subpath_is_closed,
};

pub(crate) fn remove_overlaps(outline: &BezPath, verify: bool, verify_size: u32) -> BezPath {
    let Some(simplified) = simplify(outline) else {
        return outline.clone();
    };
    if verify && !renders_match(outline, &simplified, verify_size) {
        return outline.clone();
    }
    simplified
}

pub(crate) fn random_remove_overlaps(
    outline: &BezPath,
    random_values: &[f32],
    verify: bool,
    verify_size: u32,
) -> BezPath {
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
            let chosen = simplify(&component).unwrap_or(component);
            result.extend(chosen.elements().iter().copied());
        } else {
            for &other in group {
                result.extend(source[other].iter().copied());
            }
        }
    }
    if verify && !renders_match(outline, &result, verify_size) {
        outline.clone()
    } else {
        result
    }
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

fn renders_match(original: &BezPath, candidate: &BezPath, size: u32) -> bool {
    let Some(bounds) = combined_bounds(original, candidate) else {
        return true;
    };
    render_coverage(original, bounds, size) == render_coverage(candidate, bounds, size)
}

fn combined_bounds(first: &BezPath, second: &BezPath) -> Option<Bounds> {
    match (bounds_from_outline(first), bounds_from_outline(second)) {
        (Some(first), Some(second)) => Some(Bounds {
            x_min: first.x_min.min(second.x_min),
            y_min: first.y_min.min(second.y_min),
            x_max: first.x_max.max(second.x_max),
            y_max: first.y_max.max(second.y_max),
        }),
        (bounds @ Some(_), None) | (None, bounds @ Some(_)) => bounds,
        (None, None) => None,
    }
}

fn render_coverage(outline: &BezPath, bounds: Bounds, size: u32) -> Vec<u8> {
    render_bitmap_in_bounds(outline, size, bounds, PathFillType::Winding, false).data
}

fn simplify(outline: &BezPath) -> Option<BezPath> {
    let path = build_skia_path(outline)?;
    let scaled = path.try_make_scale((PATHOPS_SCALE, PATHOPS_SCALE))?;
    let simplified = scaled.simplify()?;

    // Simplify emits an even-odd path. Reorient nested contours before
    // returning to TorchFont, whose outlines use non-zero winding semantics.
    let simplified = outline_from_path(&simplified)?;
    let mut winding = winding_from_even_odd(&simplified);
    winding.apply_affine(kurbo::Affine::scale(f64::from(PATHOPS_SCALE.recip())));
    Some(winding)
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
        outline.extend(elements.drain(..));
        if closed {
            outline.close_path();
        }
    }
}

fn point(point: skia_safe::Point) -> Point {
    Point::new(point.x.into(), point.y.into())
}

#[cfg(test)]
mod tests {
    use kurbo::{BezPath, Rect, Shape};
    use skia_safe::{PathBuilder, PathFillType};

    use super::outline_from_path;

    fn rectangle(rect: Rect) -> BezPath {
        rect.to_path(0.1)
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
    fn renders_match_identifies_identical_font_unit_outlines() {
        let square = rectangle(Rect::new(100.0, 200.0, 900.0, 1_000.0));

        assert!(super::renders_match(&square, &square, 128));
    }

    #[test]
    fn renders_match_detects_different_font_unit_coverage() {
        let small = rectangle(Rect::new(100.0, 200.0, 300.0, 400.0));
        let large = rectangle(Rect::new(100.0, 200.0, 900.0, 1_000.0));

        assert!(!super::renders_match(&small, &large, 128));
    }

    #[test]
    fn renders_match_uses_the_same_bounds_for_both_outlines() {
        let left = rectangle(Rect::new(0.0, 0.0, 500.0, 1_000.0));
        let right = rectangle(Rect::new(500.0, 0.0, 1_000.0, 1_000.0));

        assert!(!super::renders_match(&left, &right, 128));
    }

    #[test]
    fn remove_overlaps_with_verify_keeps_a_correct_simplification() {
        let mut outline = BezPath::new();
        outline.move_to((100.0, 100.0));
        outline.line_to((500.0, 100.0));
        outline.line_to((500.0, 500.0));
        outline.line_to((100.0, 500.0));
        outline.close_path();
        outline.move_to((300.0, 100.0));
        outline.line_to((700.0, 100.0));
        outline.line_to((700.0, 500.0));
        outline.line_to((300.0, 500.0));
        outline.close_path();

        let unverified = super::remove_overlaps(&outline, false, 128);
        let verified = super::remove_overlaps(&outline, true, 128);

        assert_eq!(unverified.subpaths().count(), 1);
        assert_eq!(verified.subpaths().count(), 1);
        assert!(super::renders_match(&unverified, &verified, 128));
    }
}
