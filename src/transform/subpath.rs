#[cfg(test)]
use crate::outline::outline_from_subpaths;
use crate::outline::{
    BezPath, PathEl, Point, path_element_end, subpath_elements, subpath_from_elements,
    subpath_is_closed, subpath_start,
};

pub(crate) fn reverse_subpath(subpath: &[PathEl]) -> BezPath {
    BezPath::from_vec(subpath.to_vec()).reverse_subpaths()
}

pub(crate) fn split_subpaths(outline: &BezPath) -> Vec<BezPath> {
    outline
        .subpaths()
        .map(|subpath| BezPath::from_vec(subpath.to_vec()))
        .collect()
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

    let start = path_element_end(source[start_idx - 1]);
    let mut elements = Vec::with_capacity(source.len() + 1);
    elements.extend_from_slice(&source[start_idx..]);
    if source
        .last()
        .is_none_or(|element| path_element_end(*element) != subpath_start(subpath))
    {
        elements.push(PathEl::LineTo(subpath_start(subpath)));
    }
    elements.extend_from_slice(&source[..start_idx]);
    subpath_from_elements(start, elements, true)
}

fn nodes(subpath: &[PathEl]) -> impl Iterator<Item = Point> + '_ {
    std::iter::once(subpath_start(subpath)).chain(
        subpath_elements(subpath)
            .iter()
            .map(|element| path_element_end(*element)),
    )
}

fn compare_points(a: Point, b: Point) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::PathEl;

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

    // --- reverse_subpath ---

    #[test]
    fn splits_subpaths_in_order() {
        let first = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let second = closed(pt(2.0, 0.0), vec![line(3.0, 0.0)]);
        let outline = outline_from_subpaths([first.clone(), second.clone()]);

        assert_eq!(split_subpaths(&outline), [first, second]);
    }

    #[test]
    fn reverse_subpath_empty_returns_clone() {
        let s = open(pt(0.0, 0.0), vec![]);
        let r = reverse_subpath(s.elements());
        assert_eq!(r, s);
    }

    #[test]
    fn reverse_subpath_triangle() {
        // A→B→C  reversed to  C→B→A
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

    // --- normalize_subpath_start_points ---

    #[test]
    fn normalize_picks_lexicographic_minimum() {
        // Triangle B(2,0)→C(0,0)→A(1,1), start=B
        // Lex minimum is C(0,0) at index 1
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
    fn normalize_already_minimum_is_noop() {
        // A(0,0) is already the minimum
        let subpath = closed(pt(0.0, 0.0), vec![line(1.0, 0.0), line(0.5, 1.0)]);
        let outline = outline_from_subpaths([subpath.clone()]);
        let result = normalize_subpath_start_points(&outline);
        assert_eq!(subpaths(&result)[0], subpath);
    }

    #[test]
    fn normalize_prefers_smaller_index_for_equal_points() {
        // Explicitly closed: A→B→A; nodes=[A,B,A]. Both A's tie — index 0 must win.
        let outline =
            outline_from_subpaths([closed(pt(0.0, 0.0), vec![line(1.0, 0.0), line(0.0, 0.0)])]);
        let result = normalize_subpath_start_points(&outline);
        let result_subpaths = subpaths(&result);
        let s = &result_subpaths[0];
        assert_eq!(subpath_start(s.elements()), pt(0.0, 0.0));
        assert_eq!(subpath_elements(s.elements()).len(), 2);
    }

    #[test]
    fn normalize_does_not_rotate_open_subpath() {
        let subpath = open(pt(2.0, 0.0), vec![line(0.0, 0.0), line(1.0, 1.0)]);
        let outline = outline_from_subpaths([subpath.clone()]);
        let result = normalize_subpath_start_points(&outline);
        assert_eq!(subpaths(&result)[0], subpath);
    }

    // --- randomize_subpath_start_points ---

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

    // --- reverse_closed_subpaths ---

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
}
