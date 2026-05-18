use crate::outline::{Outline, PathElement, Point};

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) x_min: f32,
    pub(crate) y_min: f32,
    pub(crate) x_max: f32,
    pub(crate) y_max: f32,
}

impl Bounds {
    pub(crate) fn new(point: Point) -> Self {
        Self {
            x_min: point.x,
            y_min: point.y,
            x_max: point.x,
            y_max: point.y,
        }
    }

    pub(crate) fn include(&mut self, point: Point) {
        self.x_min = self.x_min.min(point.x);
        self.y_min = self.y_min.min(point.y);
        self.x_max = self.x_max.max(point.x);
        self.y_max = self.y_max.max(point.y);
    }

    pub(crate) fn width(self) -> f32 {
        self.x_max - self.x_min
    }

    pub(crate) fn height(self) -> f32 {
        self.y_max - self.y_min
    }
}

#[derive(Default)]
pub(crate) struct BoundsPen {
    bounds: Option<Bounds>,
    current: Option<Point>,
    subpath_start: Option<Point>,
}

impl BoundsPen {
    pub(crate) fn finish(self) -> Option<Bounds> {
        self.bounds
    }

    pub(crate) fn move_to(&mut self, point: Point) {
        self.include(point);
        self.current = Some(point);
        self.subpath_start = Some(point);
    }

    pub(crate) fn line_to(&mut self, point: Point) {
        self.include(point);
        self.current = Some(point);
    }

    pub(crate) fn quad_to(&mut self, control: Point, end: Point) {
        let start = self.current.unwrap_or(end);
        self.include_quad(start, control, end);
        self.current = Some(end);
    }

    pub(crate) fn curve_to(&mut self, control0: Point, control1: Point, end: Point) {
        let start = self.current.unwrap_or(end);
        self.include_cubic(start, control0, control1, end);
        self.current = Some(end);
    }

    pub(crate) fn close(&mut self) {
        self.current = self.subpath_start;
    }

    pub(crate) fn path_element(&mut self, element: PathElement) {
        match element {
            PathElement::LineTo(point) => self.line_to(point),
            PathElement::QuadTo { control, end } => self.quad_to(control, end),
            PathElement::CurveTo {
                control0,
                control1,
                end,
            } => self.curve_to(control0, control1, end),
        }
    }

    fn include(&mut self, point: Point) {
        if let Some(bounds) = &mut self.bounds {
            bounds.include(point);
        } else {
            self.bounds = Some(Bounds::new(point));
        }
    }

    fn include_quad(&mut self, start: Point, control: Point, end: Point) {
        self.include(end);
        for t in [
            quad_extremum(start.x, control.x, end.x),
            quad_extremum(start.y, control.y, end.y),
        ]
        .into_iter()
        .flatten()
        {
            self.include(Point::new(
                quad_at(start.x, control.x, end.x, t),
                quad_at(start.y, control.y, end.y, t),
            ));
        }
    }

    fn include_cubic(&mut self, start: Point, control0: Point, control1: Point, end: Point) {
        self.include(end);
        for t in cubic_extrema(start.x, control0.x, control1.x, end.x)
            .into_iter()
            .flatten()
            .chain(
                cubic_extrema(start.y, control0.y, control1.y, end.y)
                    .into_iter()
                    .flatten(),
            )
        {
            self.include(Point::new(
                cubic_at(start.x, control0.x, control1.x, end.x, t),
                cubic_at(start.y, control0.y, control1.y, end.y, t),
            ));
        }
    }
}

pub(crate) fn bounds_from_outline(outline: &Outline) -> Option<Bounds> {
    let mut pen = BoundsPen::default();
    for subpath in outline.subpaths() {
        let start = subpath.start();
        pen.move_to(start);
        for element in subpath.elements() {
            pen.path_element(*element);
        }
        if subpath.is_closed() {
            pen.close();
        }
    }
    pen.finish()
}

fn quad_extremum(p0: f32, p1: f32, p2: f32) -> Option<f32> {
    let denom = p0 - 2.0 * p1 + p2;
    if denom.abs() <= f32::EPSILON {
        return None;
    }
    let t = (p0 - p1) / denom;
    (t > 0.0 && t < 1.0).then_some(t)
}

fn cubic_extrema(p0: f32, p1: f32, p2: f32, p3: f32) -> [Option<f32>; 2] {
    let a = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
    let b = 2.0 * (p0 - 2.0 * p1 + p2);
    let c = p1 - p0;
    solve_quadratic(a, b, c).map(|t| t.filter(|t| *t > 0.0 && *t < 1.0))
}

fn solve_quadratic(a: f32, b: f32, c: f32) -> [Option<f32>; 2] {
    if a.abs() <= f32::EPSILON {
        if b.abs() <= f32::EPSILON {
            return [None, None];
        }
        return [Some(-c / b), None];
    }
    let disc = b.mul_add(b, -4.0 * a * c);
    if disc < 0.0 {
        return [None, None];
    }
    if disc <= f32::EPSILON {
        return [Some(-b / (2.0 * a)), None];
    }
    let root = disc.sqrt();
    [Some((-b + root) / (2.0 * a)), Some((-b - root) / (2.0 * a))]
}

fn quad_at(p0: f32, p1: f32, p2: f32, t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * p0 + 2.0 * mt * t * p1 + t * t * p2
}

fn cubic_at(p0: f32, p1: f32, p2: f32, p3: f32, t: f32) -> f32 {
    let mt = 1.0 - t;
    mt * mt * mt * p0 + 3.0 * mt * mt * t * p1 + 3.0 * mt * t * t * p2 + t * t * t * p3
}

#[cfg(test)]
mod tests {
    use super::*;

    fn close_enough(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn computes_quadratic_tight_bounds() {
        let mut pen = BoundsPen::default();
        pen.move_to(Point::new(0.0, 0.0));
        pen.quad_to(Point::new(1.0, 2.0), Point::new(2.0, 0.0));

        let bounds = pen.finish().unwrap();

        close_enough(bounds.x_min, 0.0);
        close_enough(bounds.y_min, 0.0);
        close_enough(bounds.x_max, 2.0);
        close_enough(bounds.y_max, 1.0);
    }

    #[test]
    fn computes_cubic_tight_bounds() {
        let mut pen = BoundsPen::default();
        pen.move_to(Point::new(0.0, 0.0));
        pen.curve_to(
            Point::new(0.0, 3.0),
            Point::new(3.0, 3.0),
            Point::new(3.0, 0.0),
        );

        let bounds = pen.finish().unwrap();

        close_enough(bounds.x_min, 0.0);
        close_enough(bounds.y_min, 0.0);
        close_enough(bounds.x_max, 3.0);
        close_enough(bounds.y_max, 2.25);
    }
}
