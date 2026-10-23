mod encoding;

use kurbo::{PathEl, PathSeg, Point, segments};

pub(crate) use encoding::{DecodeError, ElementType, decode, encode};

pub(crate) fn subpath_start(subpath: &[PathEl]) -> Point {
    let Some(PathEl::MoveTo(start)) = subpath.first() else {
        unreachable!("kurbo subpaths must start with MoveTo");
    };
    *start
}

pub(crate) fn subpath_is_closed(subpath: &[PathEl]) -> bool {
    matches!(subpath.last(), Some(PathEl::ClosePath))
}

pub(crate) fn subpath_segments(subpath: &[PathEl]) -> impl Iterator<Item = PathSeg> + '_ {
    segments(
        subpath
            .iter()
            .copied()
            .filter(|&element| element != PathEl::ClosePath),
    )
}
