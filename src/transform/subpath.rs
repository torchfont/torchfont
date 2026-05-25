use crate::geom::{Outline, PathElement, Point, Subpath, reverse_closed_subpath, subpath_nodes};

pub(crate) fn normalize_subpath_start_points(outline: &Outline) -> Outline {
    transform_start_points(outline, |subpath, _| {
        subpath_nodes(subpath)
            .into_iter()
            .enumerate()
            .min_by(|(_, a), (_, b)| compare_points(*a, *b))
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
                reverse_closed_subpath(subpath)
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

    let nodes = subpath_nodes(subpath);
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
