use kurbo::Shape;

use super::{BezPath, PathEl, Point, subpath_start};

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
            x_min: round_down(point.x),
            y_min: round_down(point.y),
            x_max: round_up(point.x),
            y_max: round_up(point.y),
        }
    }

    pub(crate) fn include(&mut self, point: Point) {
        self.x_min = self.x_min.min(round_down(point.x));
        self.y_min = self.y_min.min(round_down(point.y));
        self.x_max = self.x_max.max(round_up(point.x));
        self.y_max = self.y_max.max(round_up(point.y));
    }

    fn include_rect(&mut self, rect: kurbo::Rect) {
        self.include(Point::new(rect.x0, rect.y0));
        self.include(Point::new(rect.x1, rect.y1));
    }

    fn include_bounds(&mut self, other: Self) {
        self.x_min = self.x_min.min(other.x_min);
        self.y_min = self.y_min.min(other.y_min);
        self.x_max = self.x_max.max(other.x_max);
        self.y_max = self.y_max.max(other.y_max);
    }

    pub(crate) fn width(self) -> f32 {
        self.x_max - self.x_min
    }

    pub(crate) fn height(self) -> f32 {
        self.y_max - self.y_min
    }
}

fn round_down(value: f64) -> f32 {
    let rounded = value as f32;
    if rounded.is_finite() && f64::from(rounded) > value {
        rounded.next_down()
    } else {
        rounded
    }
}

fn round_up(value: f64) -> f32 {
    let rounded = value as f32;
    if rounded.is_finite() && f64::from(rounded) < value {
        rounded.next_up()
    } else {
        rounded
    }
}

pub(crate) fn bounds_from_outline(outline: &BezPath) -> Option<Bounds> {
    let mut subpaths = outline.subpaths();
    let mut bounds = bounds_from_subpath(subpaths.next()?);
    for subpath in subpaths {
        bounds.include_bounds(bounds_from_subpath(subpath));
    }
    Some(bounds)
}

pub(crate) fn bounds_from_subpath(subpath: &[PathEl]) -> Bounds {
    let mut bounds = Bounds::new(subpath_start(subpath));
    if !super::subpath_elements(subpath).is_empty() {
        bounds.include_rect(subpath.bounding_box());
    }
    bounds
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::outline::{PathEl, outline_from_subpaths, subpath_from_elements};

    fn close_enough(a: f32, b: f32) {
        assert!((a - b).abs() < 1e-5, "{a} != {b}");
    }

    #[test]
    fn computes_quadratic_tight_bounds() {
        let outline = outline_from_subpaths([subpath_from_elements(
            Point::new(0.0, 0.0),
            [PathEl::QuadTo(Point::new(1.0, 2.0), Point::new(2.0, 0.0))],
            false,
        )]);
        let bounds = bounds_from_outline(&outline).unwrap();

        close_enough(bounds.x_min, 0.0);
        close_enough(bounds.y_min, 0.0);
        close_enough(bounds.x_max, 2.0);
        close_enough(bounds.y_max, 1.0);
    }

    #[test]
    fn computes_cubic_tight_bounds() {
        let outline = outline_from_subpaths([subpath_from_elements(
            Point::new(0.0, 0.0),
            [PathEl::CurveTo(
                Point::new(0.0, 3.0),
                Point::new(3.0, 3.0),
                Point::new(3.0, 0.0),
            )],
            false,
        )]);
        let bounds = bounds_from_outline(&outline).unwrap();

        close_enough(bounds.x_min, 0.0);
        close_enough(bounds.y_min, 0.0);
        close_enough(bounds.x_max, 3.0);
        close_enough(bounds.y_max, 2.25);
    }

    #[test]
    fn f32_bounds_enclose_f64_coordinates() {
        let value = 1.0 + f64::EPSILON;
        let bounds = Bounds::new(Point::new(value, -value));

        assert!(f64::from(bounds.x_min) <= value);
        assert!(f64::from(bounds.x_max) >= value);
        assert!(f64::from(bounds.y_min) <= -value);
        assert!(f64::from(bounds.y_max) >= -value);
    }

    #[test]
    fn move_only_bounds_do_not_include_origin() {
        let mut outline = BezPath::new();
        outline.move_to((10.0, 20.0));

        assert_eq!(
            bounds_from_outline(&outline),
            Some(Bounds {
                x_min: 10.0,
                y_min: 20.0,
                x_max: 10.0,
                y_max: 20.0,
            })
        );
    }

    #[test]
    fn combines_drawn_and_move_only_subpaths() {
        let mut outline = BezPath::new();
        outline.move_to((10.0, 20.0));
        outline.move_to((-2.0, -3.0));
        outline.line_to((4.0, 5.0));

        assert_eq!(
            bounds_from_outline(&outline),
            Some(Bounds {
                x_min: -2.0,
                y_min: -3.0,
                x_max: 10.0,
                y_max: 20.0,
            })
        );
    }
}
