use kurbo::{BezPath, PathEl, Point, Shape};

use crate::outline::{subpath_is_closed, subpath_start};

pub(crate) fn split_subpaths(outline: &BezPath) -> Vec<BezPath> {
    outline
        .subpaths()
        .map(|subpath| BezPath::from_vec(subpath.to_vec()))
        .collect()
}

pub(crate) fn drop_subpaths(outline: &BezPath, drop_mask: &[bool]) -> BezPath {
    let mut result = BezPath::new();
    for (subpath, &drop) in outline.subpaths().zip(drop_mask) {
        if !drop {
            result.extend(subpath.iter().copied());
        }
    }
    result
}

pub(crate) fn truncate_subpaths(
    outline: &BezPath,
    max_length: Option<usize>,
    max_subpaths: Option<usize>,
) -> BezPath {
    let mut result = BezPath::new();
    let mut remaining = max_length.map(|length| length - 1);
    for (index, subpath) in outline.subpaths().enumerate() {
        if max_subpaths.is_some_and(|count| index >= count)
            || remaining.is_some_and(|length| subpath.len() > length)
        {
            break;
        }
        result.extend(subpath.iter().copied());
        if let Some(length) = &mut remaining {
            *length -= subpath.len();
        }
    }
    result
}

pub(crate) fn drop_subpaths_to_fit(
    outline: &BezPath,
    removal_values: &[f32],
    max_length: Option<usize>,
    max_subpaths: Option<usize>,
) -> BezPath {
    let subpaths: Vec<_> = outline.subpaths().collect();
    let mut element_count = subpaths.iter().map(|subpath| subpath.len()).sum::<usize>() + 1;
    let mut subpath_count = subpaths.len();
    let mut removal_order: Vec<_> = removal_values.iter().copied().enumerate().collect();
    removal_order.sort_by(|(a_idx, a), (b_idx, b)| a.total_cmp(b).then_with(|| a_idx.cmp(b_idx)));
    let mut drop_mask = vec![false; subpath_count];

    for (index, _) in removal_order {
        let exceeds_length = max_length.is_some_and(|limit| element_count > limit);
        let exceeds_count = max_subpaths.is_some_and(|limit| subpath_count > limit);
        if !exceeds_length && !exceeds_count {
            break;
        }
        drop_mask[index] = true;
        element_count -= subpaths[index].len();
        subpath_count -= 1;
    }

    drop_subpaths(outline, &drop_mask)
}

pub(crate) fn normalize_subpath_start_points(outline: &BezPath) -> BezPath {
    transform_start_points(outline, |subpath, _| {
        nodes(subpath)
            .enumerate()
            .min_by(|(i, a), (j, b)| compare_points(*a, *b).then(i.cmp(j)))
            .map_or(0, |(idx, _)| idx)
    })
}

pub(crate) fn randomize_subpath_start_points(outline: &BezPath, random_values: &[f32]) -> BezPath {
    transform_start_points(outline, |subpath, subpath_idx| {
        let elements = subpath_elements(subpath);
        let node_count = elements.len()
            + usize::from(
                elements
                    .last()
                    .is_none_or(|element| path_element_end(*element) != subpath_start(subpath)),
            );
        let value = random_values[subpath_idx].clamp(0.0, 1.0 - f32::EPSILON);
        (value * node_count as f32) as usize
    })
}

pub(crate) fn randomize_subpath_order(outline: &BezPath, random_values: &[f32]) -> BezPath {
    let mut keyed_subpaths: Vec<_> = outline
        .subpaths()
        .zip(random_values.iter().copied())
        .enumerate()
        .collect();
    keyed_subpaths
        .sort_by(|(a_idx, (_, a)), (b_idx, (_, b))| a.total_cmp(b).then_with(|| a_idx.cmp(b_idx)));
    let mut result = BezPath::new();
    for (_, (subpath, _)) in keyed_subpaths {
        result.extend(subpath.iter().copied());
    }
    result
}

pub(crate) fn normalize_subpath_order(outline: &BezPath) -> BezPath {
    let mut subpaths: Vec<_> = outline.subpaths().enumerate().collect();
    subpaths.sort_by(|(a_idx, a), (b_idx, b)| {
        let a = a.bounding_box();
        let b = b.bounding_box();
        a.x0.total_cmp(&b.x0)
            .then_with(|| a.y0.total_cmp(&b.y0))
            .then_with(|| a.x1.total_cmp(&b.x1))
            .then_with(|| a.y1.total_cmp(&b.y1))
            .then_with(|| a_idx.cmp(b_idx))
    });
    let mut result = BezPath::new();
    for (_, subpath) in subpaths {
        result.extend(subpath.iter().copied());
    }
    result
}

fn transform_start_points(
    outline: &BezPath,
    choose_start: impl Fn(&[PathEl], usize) -> usize,
) -> BezPath {
    let mut result = BezPath::new();
    for (idx, subpath) in outline.subpaths().enumerate() {
        if subpath_is_closed(subpath) {
            let rotated = rotate_closed_subpath(subpath, choose_start(subpath, idx));
            result.extend(rotated.elements().iter().copied());
        } else {
            result.extend(subpath.iter().copied());
        }
    }
    result
}

fn rotate_closed_subpath(subpath: &[PathEl], start_idx: usize) -> BezPath {
    let source = subpath_elements(subpath);
    if start_idx == 0 || source.is_empty() {
        return BezPath::from_vec(subpath.to_vec());
    }

    let mut elements = Vec::with_capacity(source.len() + 3);
    elements.push(PathEl::MoveTo(path_element_end(source[start_idx - 1])));
    elements.extend_from_slice(&source[start_idx..]);
    if source
        .last()
        .is_none_or(|element| path_element_end(*element) != subpath_start(subpath))
    {
        elements.push(PathEl::LineTo(subpath_start(subpath)));
    }
    elements.extend_from_slice(&source[..start_idx]);
    elements.push(PathEl::ClosePath);
    BezPath::from_vec(elements)
}

fn nodes(subpath: &[PathEl]) -> impl Iterator<Item = Point> + '_ {
    std::iter::once(subpath_start(subpath)).chain(
        subpath_elements(subpath)
            .iter()
            .map(|element| path_element_end(*element)),
    )
}

fn subpath_elements(subpath: &[PathEl]) -> &[PathEl] {
    &subpath[1..subpath.len() - usize::from(subpath_is_closed(subpath))]
}

fn path_element_end(element: PathEl) -> Point {
    element
        .end_point()
        .expect("drawing elements always have an end point")
}

fn compare_points(a: Point, b: Point) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn outline_from_subpaths(subpaths: impl IntoIterator<Item = BezPath>) -> BezPath {
        subpaths.into_iter().flatten().collect()
    }

    fn pt(x: f32, y: f32) -> Point {
        Point::new(x.into(), y.into())
    }

    fn line(x: f32, y: f32) -> PathEl {
        PathEl::LineTo(pt(x, y))
    }

    fn closed(start: Point, elements: Vec<PathEl>) -> BezPath {
        std::iter::once(PathEl::MoveTo(start))
            .chain(elements)
            .chain([PathEl::ClosePath])
            .collect()
    }

    fn open(start: Point, elements: Vec<PathEl>) -> BezPath {
        std::iter::once(PathEl::MoveTo(start))
            .chain(elements)
            .collect()
    }

    fn subpaths(path: &BezPath) -> Vec<BezPath> {
        path.subpaths()
            .map(|subpath| BezPath::from_vec(subpath.to_vec()))
            .collect()
    }

    #[test]
    fn drops_selected_subpaths() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(2.0, 0.0), vec![line(3.0, 0.0)]);
        let outline = outline_from_subpaths([first.clone(), second]);

        assert_eq!(drop_subpaths(&outline, &[false, true]), first);
        assert!(drop_subpaths(&outline, &[true, true]).is_empty());
    }

    #[test]
    fn truncates_at_subpath_boundaries() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(2.0, 0.0), vec![line(3.0, 0.0)]);
        let outline = outline_from_subpaths([first.clone(), second]);

        assert_eq!(truncate_subpaths(&outline, Some(4), None), first);
    }

    #[test]
    fn truncation_keeps_every_fitting_subpath() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(2.0, 0.0), vec![line(3.0, 0.0)]);
        let outline = outline_from_subpaths([first, second]);

        assert_eq!(truncate_subpaths(&outline, Some(6), None), outline);
    }

    #[test]
    fn truncation_can_return_an_empty_path() {
        let outline = outline_from_subpaths([open(pt(0.0, 0.0), vec![line(1.0, 0.0)])]);

        assert!(truncate_subpaths(&outline, Some(1), None).is_empty());
    }

    #[test]
    fn truncates_by_subpath_count() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(2.0, 0.0), vec![line(3.0, 0.0)]);
        let outline = outline_from_subpaths([first.clone(), second]);

        assert_eq!(truncate_subpaths(&outline, None, Some(1)), first);
        assert!(truncate_subpaths(&outline, None, Some(0)).is_empty());
    }

    #[test]
    fn drops_randomly_ordered_subpaths_until_limits_fit() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(2.0, 0.0), vec![line(3.0, 0.0)]);
        let outline = outline_from_subpaths([first, second.clone()]);

        assert_eq!(
            drop_subpaths_to_fit(&outline, &[0.1, 0.9], None, Some(1)),
            second
        );
    }

    #[test]
    fn random_truncation_preserves_an_outline_that_fits() {
        let outline = outline_from_subpaths([open(pt(0.0, 0.0), vec![line(1.0, 0.0)])]);

        assert_eq!(
            drop_subpaths_to_fit(&outline, &[0.1], Some(3), Some(1)),
            outline
        );
    }

    #[test]
    fn normalize_picks_lexicographic_minimum() {
        let outline =
            outline_from_subpaths([closed(pt(2.0, 0.0), vec![line(0.0, 0.0), line(1.0, 1.0)])]);
        let result = normalize_subpath_start_points(&outline);
        let result_subpaths = subpaths(&result);
        let s = &result_subpaths[0];
        assert_eq!(subpath_start(s.elements()), pt(0.0, 0.0));
        assert_eq!(subpath_elements(s.elements())[0], line(1.0, 1.0));
        assert_eq!(subpath_elements(s.elements())[1], line(2.0, 0.0));
        assert_eq!(subpath_elements(s.elements())[2], line(0.0, 0.0));
    }

    #[test]
    fn normalize_prefers_smaller_index_for_equal_points() {
        let outline = outline_from_subpaths([closed(
            pt(0.0, 0.0),
            vec![line(1.0, 0.0), line(0.0, 0.0), line(0.0, 1.0)],
        )]);
        let result = normalize_subpath_start_points(&outline);
        assert_eq!(result, outline);
    }

    #[test]
    fn randomize_counts_implicit_close_endpoint() {
        let outline =
            outline_from_subpaths([closed(pt(0.0, 0.0), vec![line(1.0, 0.0), line(1.0, 1.0)])]);
        let result = randomize_subpath_start_points(&outline, &[0.9]);

        assert_eq!(subpath_start(subpaths(&result)[0].elements()), pt(1.0, 1.0));
    }

    #[test]
    fn randomize_excludes_duplicate_explicit_close_endpoint() {
        let outline = outline_from_subpaths([closed(
            pt(0.0, 0.0),
            vec![line(1.0, 0.0), line(1.0, 1.0), line(0.0, 0.0)],
        )]);
        let result = randomize_subpath_start_points(&outline, &[0.9]);

        assert_eq!(subpath_start(subpaths(&result)[0].elements()), pt(1.0, 1.0));
    }

    #[test]
    fn randomize_subpath_order_uses_random_keys() {
        let first = closed(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(10.0, 0.0), vec![line(11.0, 0.0)]);
        let outline = outline_from_subpaths([first.clone(), second.clone()]);

        let result = randomize_subpath_order(&outline, &[0.9, 0.1]);

        assert_eq!(subpaths(&result), vec![second, first]);
    }

    #[test]
    fn randomize_subpath_order_preserves_tie_order() {
        let first = open(pt(0.0, 0.0), vec![]);
        let second = open(pt(10.0, 0.0), vec![]);
        let outline = outline_from_subpaths([first.clone(), second.clone()]);

        let result = randomize_subpath_order(&outline, &[0.5, 0.5]);

        assert_eq!(subpaths(&result), vec![first, second]);
    }

    #[test]
    fn normalize_subpath_order_sorts_bounds_lexicographically() {
        let first = closed(pt(10.0, 0.0), vec![line(11.0, 0.0)]);
        let second = closed(pt(0.0, 10.0), vec![line(1.0, 10.0)]);
        let third = closed(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let outline = outline_from_subpaths([first.clone(), second.clone(), third.clone()]);

        let result = normalize_subpath_order(&outline);

        assert_eq!(subpaths(&result), vec![third, second, first]);
    }

    #[test]
    fn normalize_subpath_order_does_not_depend_on_start_point_or_winding() {
        let right = closed(pt(10.0, 0.0), vec![line(12.0, 0.0), line(11.0, 1.0)]);
        let left = closed(pt(2.0, 0.0), vec![line(0.0, 0.0), line(1.0, 1.0)]);
        let outline = outline_from_subpaths([right.clone(), left.clone()]);
        let changed = outline_from_subpaths([right.reverse_subpaths(), left.clone()]);

        assert_eq!(
            subpaths(&normalize_subpath_order(&outline)),
            vec![left.clone(), right.clone()]
        );
        assert_eq!(
            subpaths(&normalize_subpath_order(&changed)),
            vec![left, right.reverse_subpaths()]
        );
    }

    #[test]
    fn normalize_subpath_order_preserves_tie_order() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 1.0)]);
        let second = open(
            pt(0.0, 0.0),
            vec![PathEl::QuadTo(pt(0.5, 0.5), pt(1.0, 1.0))],
        );
        let outline = outline_from_subpaths([first.clone(), second.clone()]);

        let result = normalize_subpath_order(&outline);

        assert_eq!(subpaths(&result), vec![first, second]);
    }
}
