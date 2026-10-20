use skia_safe::PathFillType;

use super::overlap::{bounds_groups, overlap_groups, subpath_area};
use super::render_bitmap::render_bitmap_in_bounds;
use super::skia::{build_skia_path, outline_from_path};
use crate::outline::{BezPath, Bounds, PathEl, bounds_from_outline, subpath_is_closed};

pub(crate) fn remove_overlaps(outline: &BezPath, verify: bool, verify_size: u32) -> BezPath {
    let Some(simplified) = simplify_bounds_groups(outline) else {
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
    let paths: Vec<_> = outline.subpaths().collect();
    let (indices, closed): (Vec<_>, Vec<_>) = paths
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, path)| subpath_is_closed(path))
        .unzip();
    let areas: Vec<_> = closed.iter().map(|path| subpath_area(path)).collect();
    let mut groups: Vec<Vec<_>> = overlap_groups(&closed, &areas)
        .into_iter()
        .map(|group| group.into_iter().map(|index| indices[index]).collect())
        .collect();
    groups.extend(
        (0..paths.len())
            .filter(|&index| !subpath_is_closed(paths[index]))
            .map(|index| vec![index]),
    );
    groups.sort_unstable_by_key(|group| group[0]);

    let overlapping: Vec<_> = (0..groups.len())
        .filter(|&index| groups[index].len() > 1)
        .collect();
    if overlapping.is_empty() {
        return outline.clone();
    }

    let values = &random_values[..overlapping.len()];
    let mut selected = vec![false; groups.len()];
    for (&index, &value) in overlapping.iter().zip(values) {
        selected[index] = value < 0.5;
    }
    if !selected.contains(&true) {
        let index = values
            .iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| a.total_cmp(b))
            .map_or(0, |(index, _)| index);
        selected[overlapping[index]] = true;
    }

    let result = groups
        .iter()
        .zip(selected)
        .flat_map(|(group, selected)| {
            let component = component(&paths, group);
            if selected {
                simplify(&component).unwrap_or(component)
            } else {
                component
            }
        })
        .collect();
    if verify && !renders_match(outline, &result, verify_size) {
        outline.clone()
    } else {
        result
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

fn simplify_bounds_groups(outline: &BezPath) -> Option<BezPath> {
    let paths: Vec<_> = outline.subpaths().collect();
    let groups = bounds_groups(&paths, true, |_, _| true);
    let mut result = BezPath::new();
    for group in &groups {
        result.extend(simplify(&component(&paths, group))?);
    }
    Some(result)
}

fn component(paths: &[&[PathEl]], group: &[usize]) -> BezPath {
    group
        .iter()
        .flat_map(|&index| paths[index].iter().copied())
        .collect()
}

fn simplify(outline: &BezPath) -> Option<BezPath> {
    let path = build_skia_path(outline.elements())?;
    outline_from_path(&path.simplify()?.as_winding()?)
}

#[cfg(test)]
mod tests {
    use kurbo::{BezPath, Rect, Shape};
    use skia_safe::{PathBuilder, PathFillType};

    use crate::transform::skia::{SKIA_SCALE, outline_from_path};

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
        let path = builder
            .detach()
            .try_make_scale((SKIA_SCALE, SKIA_SCALE))
            .unwrap();

        let outline = outline_from_path(&path).expect("path has real segments");
        let subpath_count = outline.subpaths().count();

        assert_eq!(subpath_count, 1);
        assert_eq!(outline.bounding_box(), Rect::new(0.0, 0.0, 10.0, 10.0));
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

    #[test]
    fn remove_overlaps_merges_touching_contours_and_keeps_disjoint_ones() {
        let mut outline = rectangle(Rect::new(0.0, 0.0, 100.0, 100.0));
        outline.extend(rectangle(Rect::new(100.0, 0.0, 200.0, 100.0)));
        outline.extend(rectangle(Rect::new(100.0, 100.0, 200.0, 200.0)));
        outline.extend(rectangle(Rect::new(300.0, 0.0, 400.0, 100.0)));

        let simplified = super::remove_overlaps(&outline, false, 128);

        assert_eq!(simplified.subpaths().count(), 2);
        assert!(super::renders_match(&outline, &simplified, 128));
    }

    #[test]
    fn simplify_preserves_nested_holes_islands_and_disjoint_contours() {
        let mut outline = rectangle(Rect::new(0.0, 0.0, 100.0, 100.0));
        outline.extend(rectangle(Rect::new(20.0, 20.0, 80.0, 80.0)).reverse_subpaths());
        outline.extend(rectangle(Rect::new(40.0, 40.0, 60.0, 60.0)));
        outline.extend(rectangle(Rect::new(120.0, 0.0, 140.0, 20.0)));

        let simplified = super::simplify(&outline).expect("valid nested contours");

        for (point, filled) in [
            ((-10.0, 50.0), false),
            ((10.0, 50.0), true),
            ((30.0, 50.0), false),
            ((50.0, 50.0), true),
            ((110.0, 10.0), false),
            ((130.0, 10.0), true),
        ] {
            assert_eq!(outline.winding(point.into()) != 0, filled);
            assert_eq!(simplified.winding(point.into()) != 0, filled);
        }
    }
}
