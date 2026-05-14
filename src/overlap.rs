use flo_curves::Coord2;
use flo_curves::bezier::path::{
    BezierPathBuilder, SimpleBezierPath, path_add, path_contains_point,
};

use crate::outline::Command;

const ACCURACY: f64 = 0.01;

#[derive(Debug)]
pub(crate) enum RemoveOverlapsError {
    InvalidShape,
}

pub(crate) fn remove_overlaps(
    types: &[i64],
    coords: &[f32],
) -> Result<(Vec<i64>, Vec<f32>), RemoveOverlapsError> {
    if types.len() * 6 != coords.len() {
        return Err(RemoveOverlapsError::InvalidShape);
    }

    let types_owned = types.to_vec();
    let coords_owned = coords.to_vec();

    let result = std::panic::catch_unwind(|| try_remove_overlaps(&types_owned, &coords_owned));

    Ok(result.unwrap_or((types_owned, coords_owned)))
}

fn try_remove_overlaps(types: &[i64], coords: &[f32]) -> (Vec<i64>, Vec<f32>) {
    if let Some(false) = contours_may_overlap(types, coords) {
        return (types.to_vec(), coords.to_vec());
    }

    let paths = outline_to_paths(types, coords);
    if paths.is_empty() {
        return end_outline();
    }

    match union_paths(paths) {
        Some(merged) => paths_to_outline(&merged),
        None => (types.to_vec(), coords.to_vec()),
    }
}

#[derive(Clone, Copy)]
struct Bounds {
    min_x: f32,
    min_y: f32,
    max_x: f32,
    max_y: f32,
}

impl Bounds {
    fn new(x: f32, y: f32) -> Self {
        Self {
            min_x: x,
            min_y: y,
            max_x: x,
            max_y: y,
        }
    }

    fn include(&mut self, x: f32, y: f32) {
        self.min_x = self.min_x.min(x);
        self.min_y = self.min_y.min(y);
        self.max_x = self.max_x.max(x);
        self.max_y = self.max_y.max(y);
    }

    fn intersects(self, other: Self) -> bool {
        self.min_x <= other.max_x
            && other.min_x <= self.max_x
            && self.min_y <= other.max_y
            && other.min_y <= self.max_y
    }

    fn contains(self, other: Self) -> bool {
        self.min_x <= other.min_x
            && other.max_x <= self.max_x
            && self.min_y <= other.min_y
            && other.max_y <= self.max_y
    }
}

struct Contour {
    bounds: Bounds,
    path: SimpleBezierPath,
    first: Coord2,
    signed_area: f32,
}

fn contours_may_overlap(types: &[i64], coords: &[f32]) -> Option<bool> {
    let contours = collect_contours(types, coords)?;

    if contours.len() <= 1 {
        return Some(false);
    }

    for (index, first) in contours.iter().enumerate() {
        if contours[index + 1..]
            .iter()
            .any(|second| contours_need_union(first, second))
        {
            return Some(true);
        }
    }
    Some(false)
}

fn collect_contours(types: &[i64], coords: &[f32]) -> Option<Vec<Contour>> {
    let mut contours = Vec::new();
    let mut current: Option<ContourBuilder> = None;

    for (&command, values) in types.iter().zip(coords.chunks_exact(6)) {
        match command {
            v if v == Command::MoveTo as i64 => {
                finish_contour(&mut contours, &mut current);
                current = Some(ContourBuilder::new(values[4], values[5]));
            }
            v if v == Command::LineTo as i64 => {
                current.as_mut()?.line_to(values[4], values[5]);
            }
            v if v == Command::QuadTo as i64 => {
                current
                    .as_mut()?
                    .quad_to(values[0], values[1], values[4], values[5]);
            }
            v if v == Command::CurveTo as i64 => {
                current.as_mut()?.cubic_to(
                    values[0], values[1], values[2], values[3], values[4], values[5],
                );
            }
            v if v == Command::Close as i64 => {
                finish_contour(&mut contours, &mut current);
            }
            v if v == Command::End as i64 => break,
            _ => return None,
        }
    }
    finish_contour(&mut contours, &mut current);

    Some(contours)
}

struct ContourBuilder {
    bounds: Bounds,
    start: (f32, f32),
    last: (f32, f32),
    segments: Vec<(Coord2, Coord2, Coord2)>,
    signed_area: f32,
}

impl ContourBuilder {
    fn new(x: f32, y: f32) -> Self {
        Self {
            bounds: Bounds::new(x, y),
            start: (x, y),
            last: (x, y),
            segments: Vec::new(),
            signed_area: 0.0,
        }
    }

    fn include_endpoint(&mut self, x: f32, y: f32) {
        self.bounds.include(x, y);
        self.signed_area += self.last.0 * y - x * self.last.1;
        self.last = (x, y);
    }

    fn line_to(&mut self, x: f32, y: f32) {
        let (lx, ly) = (self.last.0 as f64, self.last.1 as f64);
        let (ex, ey) = (x as f64, y as f64);
        self.segments.push((
            Coord2(lx + (ex - lx) / 3.0, ly + (ey - ly) / 3.0),
            Coord2(lx + 2.0 * (ex - lx) / 3.0, ly + 2.0 * (ey - ly) / 3.0),
            Coord2(ex, ey),
        ));
        self.include_endpoint(x, y);
    }

    fn quad_to(&mut self, cx: f32, cy: f32, x: f32, y: f32) {
        let (lx, ly) = (self.last.0 as f64, self.last.1 as f64);
        let (cx64, cy64) = (cx as f64, cy as f64);
        let (ex, ey) = (x as f64, y as f64);
        self.segments.push((
            Coord2(lx + 2.0 / 3.0 * (cx64 - lx), ly + 2.0 / 3.0 * (cy64 - ly)),
            Coord2(ex + 2.0 / 3.0 * (cx64 - ex), ey + 2.0 / 3.0 * (cy64 - ey)),
            Coord2(ex, ey),
        ));
        self.bounds.include(cx, cy);
        self.include_endpoint(x, y);
    }

    fn cubic_to(&mut self, cx0: f32, cy0: f32, cx1: f32, cy1: f32, x: f32, y: f32) {
        self.segments.push((
            Coord2(cx0 as f64, cy0 as f64),
            Coord2(cx1 as f64, cy1 as f64),
            Coord2(x as f64, y as f64),
        ));
        self.bounds.include(cx0, cy0);
        self.bounds.include(cx1, cy1);
        self.include_endpoint(x, y);
    }

    fn finish(self) -> Option<Contour> {
        if self.segments.is_empty() {
            return None;
        }
        let (sx, sy) = self.start;
        let signed_area = self.signed_area + self.last.0 * sy - sx * self.last.1;
        let first = Coord2(sx as f64, sy as f64);
        Some(Contour {
            bounds: self.bounds,
            path: (first, self.segments),
            first,
            signed_area,
        })
    }
}

fn finish_contour(contours: &mut Vec<Contour>, current: &mut Option<ContourBuilder>) {
    if let Some(builder) = current.take()
        && let Some(contour) = builder.finish()
    {
        contours.push(contour);
    }
}

fn contours_need_union(first: &Contour, second: &Contour) -> bool {
    if !first.bounds.intersects(second.bounds) {
        return false;
    }
    if first.signed_area.signum() != second.signed_area.signum() {
        if first.bounds.contains(second.bounds) && path_contains_point(&first.path, &second.first) {
            return false;
        }
        if second.bounds.contains(first.bounds) && path_contains_point(&second.path, &first.first) {
            return false;
        }
    }
    true
}

fn union_paths(paths: Vec<SimpleBezierPath>) -> Option<Vec<SimpleBezierPath>> {
    std::panic::catch_unwind(|| {
        let mut result = vec![paths[0].clone()];
        for path in &paths[1..] {
            result = path_add::<SimpleBezierPath>(&result, &vec![path.clone()], ACCURACY);
        }
        result
    })
    .ok()
}

fn outline_to_paths(types: &[i64], coords: &[f32]) -> Vec<SimpleBezierPath> {
    let mut paths = Vec::new();
    let mut builder: Option<BezierPathBuilder<SimpleBezierPath>> = None;
    let mut last = Coord2(0.0, 0.0);

    for (&command, values) in types.iter().zip(coords.chunks_exact(6)) {
        match command {
            v if v == Command::MoveTo as i64 => {
                if let Some(b) = builder.take() {
                    let path = b.build();
                    if !path.1.is_empty() {
                        paths.push(path);
                    }
                }
                let start = Coord2(values[4] as f64, values[5] as f64);
                builder = Some(BezierPathBuilder::<SimpleBezierPath>::start(start));
                last = start;
            }
            v if v == Command::LineTo as i64 => {
                if let Some(b) = builder.take() {
                    let end = Coord2(values[4] as f64, values[5] as f64);
                    builder = Some(b.line_to(end));
                    last = end;
                }
            }
            v if v == Command::QuadTo as i64 => {
                if let Some(b) = builder.take() {
                    let ctrl = Coord2(values[0] as f64, values[1] as f64);
                    let end = Coord2(values[4] as f64, values[5] as f64);
                    let cp1 = Coord2(
                        last.0 + 2.0 / 3.0 * (ctrl.0 - last.0),
                        last.1 + 2.0 / 3.0 * (ctrl.1 - last.1),
                    );
                    let cp2 = Coord2(
                        end.0 + 2.0 / 3.0 * (ctrl.0 - end.0),
                        end.1 + 2.0 / 3.0 * (ctrl.1 - end.1),
                    );
                    builder = Some(b.curve_to((cp1, cp2), end));
                    last = end;
                }
            }
            v if v == Command::CurveTo as i64 => {
                if let Some(b) = builder.take() {
                    let cp1 = Coord2(values[0] as f64, values[1] as f64);
                    let cp2 = Coord2(values[2] as f64, values[3] as f64);
                    let end = Coord2(values[4] as f64, values[5] as f64);
                    builder = Some(b.curve_to((cp1, cp2), end));
                    last = end;
                }
            }
            v if v == Command::Close as i64 => {
                if let Some(b) = builder.take() {
                    let path = b.build();
                    if !path.1.is_empty() {
                        paths.push(path);
                    }
                }
            }
            _ => break,
        }
    }

    if let Some(b) = builder.take() {
        let path = b.build();
        if !path.1.is_empty() {
            paths.push(path);
        }
    }

    paths
}

fn paths_to_outline(paths: &[SimpleBezierPath]) -> (Vec<i64>, Vec<f32>) {
    let mut types = Vec::new();
    let mut coords = Vec::new();

    for (start, segments) in paths {
        types.push(Command::MoveTo as i64);
        coords.extend_from_slice(&[0.0, 0.0, 0.0, 0.0, start.0 as f32, start.1 as f32]);

        for &(cp1, cp2, end) in segments {
            types.push(Command::CurveTo as i64);
            coords.extend_from_slice(&[
                cp1.0 as f32,
                cp1.1 as f32,
                cp2.0 as f32,
                cp2.1 as f32,
                end.0 as f32,
                end.1 as f32,
            ]);
        }

        types.push(Command::Close as i64);
        coords.extend_from_slice(&[0.0; 6]);
    }

    types.push(Command::End as i64);
    coords.extend_from_slice(&[0.0; 6]);

    (types, coords)
}

fn end_outline() -> (Vec<i64>, Vec<f32>) {
    (vec![Command::End as i64], vec![0.0; 6])
}
