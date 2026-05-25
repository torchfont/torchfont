use super::{Point, Subpath};

pub(crate) fn reverse_closed_subpath(subpath: &Subpath) -> Subpath {
    let Some(last) = subpath.elements().last() else {
        return subpath.clone();
    };

    let nodes = subpath_nodes(subpath);
    let elements = subpath
        .elements()
        .iter()
        .enumerate()
        .rev()
        .map(|(idx, element)| element.reversed_to(nodes[idx]))
        .collect();
    Subpath::new(last.end(), elements, true)
}

pub(crate) fn subpath_nodes(subpath: &Subpath) -> Vec<Point> {
    std::iter::once(subpath.start())
        .chain(subpath.elements().iter().map(|element| element.end()))
        .collect()
}
