use crate::geom::{Outline, PathElement, Point, Subpath};

pub(crate) fn reverse_subpath(subpath: &Subpath) -> Subpath {
    let Some(last) = subpath.elements().last() else {
        return subpath.clone();
    };
    let nodes = subpath.nodes();
    let elements = subpath
        .elements()
        .iter()
        .enumerate()
        .rev()
        .map(|(idx, element)| element.reversed_to(nodes[idx]))
        .collect();
    Subpath::new(last.end(), elements, subpath.is_closed())
}

pub(crate) fn normalize_subpath_start_points(outline: &Outline) -> Outline {
    transform_start_points(outline, |subpath, _| {
        subpath
            .nodes()
            .into_iter()
            .enumerate()
            .min_by(|(i, a), (j, b)| compare_points(*a, *b).then(i.cmp(j)))
            .map_or(0, |(idx, _)| idx)
    })
}

pub(crate) fn randomize_subpath_start_points(outline: &Outline, random_values: &[f32]) -> Outline {
    transform_start_points(outline, |subpath, subpath_idx| {
        let node_count = subpath.elements().len() + 1;
        let value = random_values[subpath_idx].clamp(0.0, 1.0 - f32::EPSILON);
        (value * node_count as f32) as usize
    })
}

pub(crate) fn reverse_closed_subpaths(outline: &Outline) -> Outline {
    let subpaths = outline
        .subpaths()
        .iter()
        .map(|subpath| {
            if subpath.is_closed() {
                reverse_subpath(subpath)
            } else {
                subpath.clone()
            }
        })
        .collect();
    Outline::new(subpaths)
}

fn transform_start_points(
    outline: &Outline,
    choose_start: impl Fn(&Subpath, usize) -> usize,
) -> Outline {
    let subpaths = outline
        .subpaths()
        .iter()
        .enumerate()
        .map(|(idx, subpath)| {
            if subpath.is_closed() {
                rotate_closed_subpath(subpath, choose_start(subpath, idx))
            } else {
                subpath.clone()
            }
        })
        .collect();
    Outline::new(subpaths)
}

fn rotate_closed_subpath(subpath: &Subpath, start_idx: usize) -> Subpath {
    if start_idx == 0 || subpath.elements().is_empty() {
        return subpath.clone();
    }

    let nodes = subpath.nodes();
    let start = nodes[start_idx];
    let mut elements = Vec::with_capacity(subpath.elements().len() + 1);
    elements.extend_from_slice(&subpath.elements()[start_idx..]);
    if subpath
        .elements()
        .last()
        .is_none_or(|e| e.end() != subpath.start())
    {
        elements.push(PathElement::LineTo(subpath.start()));
    }
    elements.extend_from_slice(&subpath.elements()[..start_idx]);
    Subpath::new(start, elements, true)
}

fn compare_points(a: Point, b: Point) -> std::cmp::Ordering {
    a.x.total_cmp(&b.x).then_with(|| a.y.total_cmp(&b.y))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::geom::PathElement;

    fn pt(x: f32, y: f32) -> Point {
        Point::new(x, y)
    }

    fn line(x: f32, y: f32) -> PathElement {
        PathElement::LineTo(pt(x, y))
    }

    fn closed(start: Point, elements: Vec<PathElement>) -> Subpath {
        Subpath::new(start, elements, true)
    }

    fn open(start: Point, elements: Vec<PathElement>) -> Subpath {
        Subpath::new(start, elements, false)
    }

    // --- reverse_subpath ---

    #[test]
    fn reverse_subpath_empty_returns_clone() {
        let s = open(pt(0.0, 0.0), vec![]);
        let r = reverse_subpath(&s);
        assert_eq!(r, s);
    }

    #[test]
    fn reverse_subpath_triangle() {
        // A→B→C  reversed to  C→B→A
        let s = open(pt(0.0, 0.0), vec![line(1.0, 0.0), line(0.5, 1.0)]);
        let r = reverse_subpath(&s);
        assert_eq!(r.start(), pt(0.5, 1.0));
        assert_eq!(r.elements()[0], line(1.0, 0.0));
        assert_eq!(r.elements()[1], line(0.0, 0.0));
    }

    #[test]
    fn reverse_subpath_cubic_swaps_controls() {
        let s = open(
            pt(0.0, 0.0),
            vec![PathElement::CurveTo {
                control0: pt(1.0, 2.0),
                control1: pt(3.0, 4.0),
                end: pt(5.0, 0.0),
            }],
        );
        let r = reverse_subpath(&s);
        assert_eq!(r.start(), pt(5.0, 0.0));
        assert_eq!(
            r.elements()[0],
            PathElement::CurveTo {
                control0: pt(3.0, 4.0),
                control1: pt(1.0, 2.0),
                end: pt(0.0, 0.0),
            }
        );
    }

    // --- normalize_subpath_start_points ---

    #[test]
    fn normalize_picks_lexicographic_minimum() {
        // Triangle B(2,0)→C(0,0)→A(1,1), start=B
        // Lex minimum is C(0,0) at index 1
        let outline = Outline::new(vec![closed(
            pt(2.0, 0.0),
            vec![line(0.0, 0.0), line(1.0, 1.0)],
        )]);
        let result = normalize_subpath_start_points(&outline);
        let s = &result.subpaths()[0];
        assert_eq!(s.start(), pt(0.0, 0.0));
        assert_eq!(s.elements()[0], line(1.0, 1.0));
        assert_eq!(s.elements()[1], line(2.0, 0.0));
        assert_eq!(s.elements()[2], line(0.0, 0.0));
    }

    #[test]
    fn normalize_already_minimum_is_noop() {
        // A(0,0) is already the minimum
        let subpath = closed(pt(0.0, 0.0), vec![line(1.0, 0.0), line(0.5, 1.0)]);
        let outline = Outline::new(vec![subpath.clone()]);
        let result = normalize_subpath_start_points(&outline);
        assert_eq!(result.subpaths()[0], subpath);
    }

    #[test]
    fn normalize_prefers_smaller_index_for_equal_points() {
        // Explicitly closed: A→B→A; nodes=[A,B,A]. Both A's tie — index 0 must win.
        let outline = Outline::new(vec![closed(
            pt(0.0, 0.0),
            vec![line(1.0, 0.0), line(0.0, 0.0)],
        )]);
        let result = normalize_subpath_start_points(&outline);
        let s = &result.subpaths()[0];
        assert_eq!(s.start(), pt(0.0, 0.0));
        assert_eq!(s.elements().len(), 2);
    }

    #[test]
    fn normalize_does_not_rotate_open_subpath() {
        let subpath = open(pt(2.0, 0.0), vec![line(0.0, 0.0), line(1.0, 1.0)]);
        let outline = Outline::new(vec![subpath.clone()]);
        let result = normalize_subpath_start_points(&outline);
        assert_eq!(result.subpaths()[0], subpath);
    }

    // --- reverse_closed_subpaths ---

    #[test]
    fn reverse_closed_subpaths_skips_open() {
        let subpath = open(pt(0.0, 0.0), vec![line(1.0, 0.0)]);
        let outline = Outline::new(vec![subpath.clone()]);
        let result = reverse_closed_subpaths(&outline);
        assert_eq!(result.subpaths()[0], subpath);
    }

    #[test]
    fn reverse_closed_subpaths_reverses_closed() {
        let outline = Outline::new(vec![closed(
            pt(0.0, 0.0),
            vec![line(1.0, 0.0), line(0.5, 1.0)],
        )]);
        let result = reverse_closed_subpaths(&outline);
        let s = &result.subpaths()[0];
        assert_eq!(s.start(), pt(0.5, 1.0));
        assert!(s.is_closed());
    }
}
