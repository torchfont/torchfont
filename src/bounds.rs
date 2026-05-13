use crate::outline::Command;

#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct Bounds {
    pub(crate) x_min: f32,
    pub(crate) y_min: f32,
    pub(crate) x_max: f32,
    pub(crate) y_max: f32,
}

impl Bounds {
    pub(crate) fn new(x: f32, y: f32) -> Self {
        Self {
            x_min: x,
            y_min: y,
            x_max: x,
            y_max: y,
        }
    }

    pub(crate) fn include(&mut self, x: f32, y: f32) {
        self.x_min = self.x_min.min(x);
        self.y_min = self.y_min.min(y);
        self.x_max = self.x_max.max(x);
        self.y_max = self.y_max.max(y);
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
    current: Option<(f32, f32)>,
    contour_start: Option<(f32, f32)>,
}

impl BoundsPen {
    pub(crate) fn finish(self) -> Option<Bounds> {
        self.bounds
    }

    pub(crate) fn move_to(&mut self, x: f32, y: f32) {
        self.include(x, y);
        self.current = Some((x, y));
        self.contour_start = Some((x, y));
    }

    pub(crate) fn line_to(&mut self, x: f32, y: f32) {
        self.include(x, y);
        self.current = Some((x, y));
    }

    pub(crate) fn quad_to(&mut self, cx0: f32, cy0: f32, x: f32, y: f32) {
        let p0 = self.current.unwrap_or((x, y));
        let p1 = (cx0, cy0);
        let p2 = (x, y);
        self.include_quad(p0, p1, p2);
        self.current = Some(p2);
    }

    pub(crate) fn curve_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        let p0 = self.current.unwrap_or((x, y));
        let p1 = (cx0, cy0);
        let p2 = (cx1, cy1);
        let p3 = (x, y);
        self.include_cubic(p0, p1, p2, p3);
        self.current = Some(p3);
    }

    pub(crate) fn close(&mut self) {
        self.current = self.contour_start;
    }

    fn include(&mut self, x: f32, y: f32) {
        if let Some(bounds) = &mut self.bounds {
            bounds.include(x, y);
        } else {
            self.bounds = Some(Bounds::new(x, y));
        }
    }

    fn include_quad(&mut self, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32)) {
        self.include(p2.0, p2.1);
        for t in [
            quad_extremum(p0.0, p1.0, p2.0),
            quad_extremum(p0.1, p1.1, p2.1),
        ]
        .into_iter()
        .flatten()
        {
            self.include(quad_at(p0.0, p1.0, p2.0, t), quad_at(p0.1, p1.1, p2.1, t));
        }
    }

    fn include_cubic(&mut self, p0: (f32, f32), p1: (f32, f32), p2: (f32, f32), p3: (f32, f32)) {
        self.include(p3.0, p3.1);
        for t in cubic_extrema(p0.0, p1.0, p2.0, p3.0)
            .into_iter()
            .chain(cubic_extrema(p0.1, p1.1, p2.1, p3.1))
        {
            self.include(
                cubic_at(p0.0, p1.0, p2.0, p3.0, t),
                cubic_at(p0.1, p1.1, p2.1, p3.1, t),
            );
        }
    }
}

pub(crate) fn bounds_from_i32_segments(types: &[i32], coords: &[f32]) -> Option<Bounds> {
    bounds_from_segments(types.iter().copied().map(i64::from), coords)
}

fn bounds_from_segments(types: impl Iterator<Item = i64>, coords: &[f32]) -> Option<Bounds> {
    let mut pen = BoundsPen::default();
    for (command, values) in types.zip(coords.chunks_exact(6)) {
        match command {
            v if v == Command::MoveTo as i64 => pen.move_to(values[4], values[5]),
            v if v == Command::LineTo as i64 => pen.line_to(values[4], values[5]),
            v if v == Command::QuadTo as i64 => {
                pen.quad_to(values[0], values[1], values[4], values[5])
            }
            v if v == Command::CurveTo as i64 => pen.curve_to(
                values[0], values[1], values[2], values[3], values[4], values[5],
            ),
            v if v == Command::Close as i64 => pen.close(),
            _ => break,
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

fn cubic_extrema(p0: f32, p1: f32, p2: f32, p3: f32) -> Vec<f32> {
    let a = -p0 + 3.0 * p1 - 3.0 * p2 + p3;
    let b = 2.0 * (p0 - 2.0 * p1 + p2);
    let c = p1 - p0;
    solve_quadratic(a, b, c)
        .into_iter()
        .filter(|t| *t > 0.0 && *t < 1.0)
        .collect()
}

fn solve_quadratic(a: f32, b: f32, c: f32) -> Vec<f32> {
    if a.abs() <= f32::EPSILON {
        if b.abs() <= f32::EPSILON {
            return Vec::new();
        }
        return vec![-c / b];
    }
    let disc = b.mul_add(b, -4.0 * a * c);
    if disc < 0.0 {
        return Vec::new();
    }
    if disc <= f32::EPSILON {
        return vec![-b / (2.0 * a)];
    }
    let root = disc.sqrt();
    vec![(-b + root) / (2.0 * a), (-b - root) / (2.0 * a)]
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
        pen.move_to(0.0, 0.0);
        pen.quad_to(1.0, 2.0, 2.0, 0.0);

        let bounds = pen.finish().unwrap();

        close_enough(bounds.x_min, 0.0);
        close_enough(bounds.y_min, 0.0);
        close_enough(bounds.x_max, 2.0);
        close_enough(bounds.y_max, 1.0);
    }

    #[test]
    fn computes_cubic_tight_bounds() {
        let mut pen = BoundsPen::default();
        pen.move_to(0.0, 0.0);
        pen.curve_to(0.0, 3.0, 3.0, 3.0, 3.0, 0.0);

        let bounds = pen.finish().unwrap();

        close_enough(bounds.x_min, 0.0);
        close_enough(bounds.y_min, 0.0);
        close_enough(bounds.x_max, 3.0);
        close_enough(bounds.y_max, 2.25);
    }
}
