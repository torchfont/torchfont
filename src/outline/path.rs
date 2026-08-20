use kurbo::{BezPath, CubicBez, Line, PathEl, PathSeg, Point, QuadBez};

#[cfg(test)]
pub(crate) fn outline_from_subpaths(subpaths: impl IntoIterator<Item = BezPath>) -> BezPath {
    let mut outline = BezPath::new();
    for subpath in subpaths {
        outline.extend(subpath.elements().iter().copied());
    }
    outline
}

pub(crate) fn subpath_from_elements(
    start: Point,
    elements: impl IntoIterator<Item = PathEl>,
    closed: bool,
) -> BezPath {
    let mut subpath = BezPath::new();
    subpath.move_to(start);
    subpath.extend(elements);
    if closed {
        subpath.close_path();
    }
    subpath
}

pub(crate) fn subpath_start(subpath: &[PathEl]) -> Point {
    let Some(PathEl::MoveTo(start)) = subpath.first() else {
        unreachable!("kurbo subpaths must start with MoveTo");
    };
    *start
}

pub(crate) fn subpath_elements(subpath: &[PathEl]) -> &[PathEl] {
    let closed = matches!(subpath.last(), Some(PathEl::ClosePath));
    &subpath[1..subpath.len() - usize::from(closed)]
}

pub(crate) fn subpath_is_closed(subpath: &[PathEl]) -> bool {
    matches!(subpath.last(), Some(PathEl::ClosePath))
}

pub(crate) fn path_element_end(element: PathEl) -> Point {
    element
        .end_point()
        .expect("drawing elements always have an end point")
}

pub(crate) fn path_seg(start: Point, element: PathEl) -> PathSeg {
    match element {
        PathEl::LineTo(end) => PathSeg::Line(Line::new(start, end)),
        PathEl::QuadTo(control, end) => PathSeg::Quad(QuadBez::new(start, control, end)),
        PathEl::CurveTo(control0, control1, end) => {
            PathSeg::Cubic(CubicBez::new(start, control0, control1, end))
        }
        PathEl::MoveTo(_) | PathEl::ClosePath => {
            unreachable!("subpath elements contain only drawing elements")
        }
    }
}
