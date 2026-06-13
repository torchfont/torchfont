use super::point::Point;

#[derive(Clone, Copy, Debug, PartialEq)]
#[allow(clippy::enum_variant_names)]
pub(crate) enum PathElement {
    LineTo(Point),
    QuadTo {
        control: Point,
        end: Point,
    },
    CurveTo {
        control0: Point,
        control1: Point,
        end: Point,
    },
}

impl PathElement {
    pub(crate) fn end(self) -> Point {
        match self {
            Self::LineTo(end) | Self::QuadTo { end, .. } | Self::CurveTo { end, .. } => end,
        }
    }

    pub(crate) fn reversed_to(self, end: Point) -> Self {
        match self {
            Self::LineTo(_) => Self::LineTo(end),
            Self::QuadTo { control, .. } => Self::QuadTo { control, end },
            Self::CurveTo {
                control0, control1, ..
            } => Self::CurveTo {
                control0: control1,
                control1: control0,
                end,
            },
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub(crate) struct Subpath {
    pub(super) start: Point,
    pub(super) elements: Vec<PathElement>,
    pub(super) closed: bool,
}

impl Subpath {
    pub(crate) fn new(start: Point, elements: Vec<PathElement>, closed: bool) -> Self {
        Self {
            start,
            elements,
            closed,
        }
    }

    pub(crate) fn start(&self) -> Point {
        self.start
    }
    pub(crate) fn elements(&self) -> &[PathElement] {
        &self.elements
    }
    pub(crate) fn is_closed(&self) -> bool {
        self.closed
    }

    pub(crate) fn nodes(&self) -> Vec<Point> {
        std::iter::once(self.start)
            .chain(self.elements.iter().map(|e| e.end()))
            .collect()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub(crate) struct Outline {
    pub(super) subpaths: Vec<Subpath>,
}

impl Outline {
    pub(crate) fn new(subpaths: Vec<Subpath>) -> Self {
        Self { subpaths }
    }

    pub(crate) fn subpaths(&self) -> &[Subpath] {
        &self.subpaths
    }
}

pub(crate) struct SubpathBuilder {
    pub(crate) start: Point,
    pub(crate) elements: Vec<PathElement>,
}

impl SubpathBuilder {
    pub(crate) fn new(start: Point) -> Self {
        Self {
            start,
            elements: Vec::new(),
        }
    }

    pub(crate) fn finish(self, closed: bool) -> Subpath {
        Subpath::new(self.start, self.elements, closed)
    }
}
