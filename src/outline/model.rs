#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Point {
    pub(crate) x: f32,
    pub(crate) y: f32,
}

impl Point {
    pub(crate) fn new(x: f32, y: f32) -> Self {
        Self { x, y }
    }

    pub(crate) fn zero() -> Self {
        Self::new(0.0, 0.0)
    }

    pub(crate) fn lerp(self, other: Self, t: f32) -> Self {
        Self::new(
            self.x + (other.x - self.x) * t,
            self.y + (other.y - self.y) * t,
        )
    }

    pub(crate) fn midpoint(self, other: Self) -> Self {
        self.lerp(other, 0.5)
    }

    pub(crate) fn sub(self, other: Self) -> Self {
        Self::new(self.x - other.x, self.y - other.y)
    }

    pub(crate) fn add(self, other: Self) -> Self {
        Self::new(self.x + other.x, self.y + other.y)
    }

    pub(crate) fn cross(self, other: Self) -> f32 {
        self.x * other.y - self.y * other.x
    }

    pub(crate) fn norm(self) -> f32 {
        if !self.x.is_finite() || !self.y.is_finite() {
            return f32::INFINITY;
        }
        (self.x * self.x + self.y * self.y).sqrt()
    }
}

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

    pub(crate) fn with_subpaths(&self, subpaths: Vec<Subpath>) -> Self {
        Self { subpaths }
    }
}

pub(super) struct SubpathBuilder {
    pub(super) start: Point,
    pub(super) elements: Vec<PathElement>,
}

impl SubpathBuilder {
    pub(super) fn new(start: Point) -> Self {
        Self {
            start,
            elements: Vec::new(),
        }
    }

    pub(super) fn finish(self, closed: bool) -> Subpath {
        Subpath::new(self.start, self.elements, closed)
    }
}
