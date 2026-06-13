use skia_safe::{Path, PathBuilder, PathFillType, PathVerb};

use super::subpath::reverse_subpath;
use crate::geom::{Bounds, BoundsPen, Outline, PathElement, Point, Subpath};

// TorchFont outlines are normalized to roughly em-sized coordinates. PathOps is
// more reliable at conventional font-unit magnitudes, so simplify a scaled copy.
const PATHOPS_SCALE: f32 = 131_072.0;

pub(crate) fn remove_overlaps(outline: &Outline) -> Outline {
    simplify(outline).unwrap_or_else(|| outline.clone())
}

fn simplify(outline: &Outline) -> Option<Outline> {
    let (path, _) = build_skia_path(outline.subpaths(), false, PathFillType::Winding)?;
    let scaled = path.try_make_scale((PATHOPS_SCALE, PATHOPS_SCALE))?;
    let simplified = scaled.simplify()?;

    // Simplify emits an even-odd path. Reorient nested contours before
    // returning to TorchFont, whose outlines use non-zero winding semantics.
    let simplified = outline_from_path(&simplified)?;
    let winding = winding_from_even_odd(&simplified)?;
    Some(scale_outline(&winding, PATHOPS_SCALE.recip()))
}

fn build_skia_path(
    subpaths: &[Subpath],
    track_bounds: bool,
    fill_type: PathFillType,
) -> Option<(Path, Option<Bounds>)> {
    let mut builder = PathBuilder::new_with_fill_type(fill_type);
    let mut bounds = track_bounds.then(BoundsPen::default);
    for subpath in subpaths {
        let start = subpath.start();
        if let Some(bounds) = &mut bounds {
            bounds.move_to(start);
        }
        builder.move_to((start.x, start.y));
        for element in subpath.elements() {
            if let Some(bounds) = &mut bounds {
                bounds.path_element(*element);
            }
            match *element {
                PathElement::LineTo(point) => builder.line_to((point.x, point.y)),
                PathElement::QuadTo { control, end } => {
                    builder.quad_to((control.x, control.y), (end.x, end.y))
                }
                PathElement::CurveTo {
                    control0,
                    control1,
                    end,
                } => builder.cubic_to(
                    (control0.x, control0.y),
                    (control1.x, control1.y),
                    (end.x, end.y),
                ),
            };
        }
        if subpath.is_closed() {
            if let Some(bounds) = &mut bounds {
                bounds.close();
            }
            builder.close();
        }
    }
    (!builder.is_empty()).then(|| (builder.detach(), bounds.and_then(BoundsPen::finish)))
}

fn outline_from_path(path: &Path) -> Option<Outline> {
    let mut subpaths = Vec::new();
    let mut start = None;
    let mut elements = Vec::new();

    for record in path.iter() {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => {
                commit_subpath(&mut start, &mut elements, &mut subpaths, false);
                start = Some(point(points[0]));
            }
            PathVerb::Line => elements.push(PathElement::LineTo(point(points[1]))),
            PathVerb::Quad => elements.push(PathElement::QuadTo {
                control: point(points[1]),
                end: point(points[2]),
            }),
            PathVerb::Cubic => elements.push(PathElement::CurveTo {
                control0: point(points[1]),
                control1: point(points[2]),
                end: point(points[3]),
            }),
            PathVerb::Close => {
                commit_subpath(&mut start, &mut elements, &mut subpaths, true);
            }
            PathVerb::Conic => return None,
        }
    }
    commit_subpath(&mut start, &mut elements, &mut subpaths, false);

    (!subpaths.is_empty()).then(|| Outline::new(subpaths))
}

fn commit_subpath(
    start: &mut Option<Point>,
    elements: &mut Vec<PathElement>,
    subpaths: &mut Vec<Subpath>,
    closed: bool,
) {
    if let Some(start) = start.take()
        && !elements.is_empty()
    {
        subpaths.push(Subpath::new(start, std::mem::take(elements), closed));
    }
}

fn point(point: skia_safe::Point) -> Point {
    Point::new(point.x, point.y)
}

fn winding_from_even_odd(outline: &Outline) -> Option<Outline> {
    // skia-pathops uses nesting depth instead of Skia's AsWinding operation;
    // keep that behavior while using TorchFont's native outline types.
    let mut contours: Vec<_> = outline
        .subpaths()
        .iter()
        .map(|subpath| {
            let path = subpath_path(subpath)?;
            Some((subpath_area(subpath), path, subpath))
        })
        .collect::<Option<_>>()?;
    contours.sort_by(|a, b| {
        b.0.abs()
            .partial_cmp(&a.0.abs())
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    let mut nesting = vec![0usize; contours.len()];
    for inner in 0..contours.len() {
        for outer in 0..inner {
            if path_is_inside(&contours[outer].1, &contours[inner].1) {
                nesting[inner] += 1;
            }
        }
    }

    let subpaths = contours
        .into_iter()
        .zip(nesting)
        .map(|((area, _, subpath), depth)| {
            let is_clockwise = area < 0.0;
            let is_outer = depth.is_multiple_of(2);
            if is_clockwise == is_outer {
                reverse_subpath(subpath)
            } else {
                subpath.clone()
            }
        })
        .collect();
    Some(Outline::new(subpaths))
}

fn subpath_path(subpath: &Subpath) -> Option<Path> {
    build_skia_path(std::slice::from_ref(subpath), false, PathFillType::EvenOdd)
        .map(|(path, _)| path)
}

fn path_is_inside(outer: &Path, inner: &Path) -> bool {
    if !outer
        .compute_tight_bounds()
        .intersects(inner.compute_tight_bounds())
    {
        return false;
    }

    inner.iter().all(|record| {
        let points = record.points();
        match record.verb() {
            PathVerb::Move => outer.contains(points[0]),
            PathVerb::Line => outer.contains(points[1]),
            PathVerb::Quad => outer.contains(points[2]),
            PathVerb::Cubic => outer.contains(points[3]),
            PathVerb::Close => true,
            PathVerb::Conic => false,
        }
    })
}

fn subpath_area(subpath: &Subpath) -> f64 {
    let start = subpath.start();
    let mut previous = start;
    let mut area = 0.0f64;

    for element in subpath.elements() {
        match *element {
            PathElement::LineTo(end) => {
                area -= (end.x - previous.x) as f64 * (end.y + previous.y) as f64 * 0.5;
                previous = end;
            }
            PathElement::QuadTo { control, end } => {
                let control = control - previous;
                let end_offset = end - previous;
                area -= (end_offset.x as f64 * control.y as f64
                    - control.x as f64 * end_offset.y as f64)
                    / 3.0;
                area -= (end.x - previous.x) as f64 * (end.y + previous.y) as f64 * 0.5;
                previous = end;
            }
            PathElement::CurveTo {
                control0,
                control1,
                end,
            } => {
                let control0 = control0 - previous;
                let control1 = control1 - previous;
                let end_offset = end - previous;
                let (c0x, c0y) = (control0.x as f64, control0.y as f64);
                let (c1x, c1y) = (control1.x as f64, control1.y as f64);
                let (ex, ey) = (end_offset.x as f64, end_offset.y as f64);
                area -=
                    (c0x * (-c1y - ey) + c1x * (c0y - 2.0 * ey) + ex * (c0y + 2.0 * c1y)) * 0.15;
                area -= (end.x - previous.x) as f64 * (end.y + previous.y) as f64 * 0.5;
                previous = end;
            }
        }
    }
    area - (start.x - previous.x) as f64 * (start.y + previous.y) as f64 * 0.5
}

fn scale_outline(outline: &Outline, scale: f32) -> Outline {
    Outline::new(
        outline
            .subpaths()
            .iter()
            .map(|subpath| {
                let elements = subpath
                    .elements()
                    .iter()
                    .map(|element| match *element {
                        PathElement::LineTo(end) => PathElement::LineTo(scale_point(end, scale)),
                        PathElement::QuadTo { control, end } => PathElement::QuadTo {
                            control: scale_point(control, scale),
                            end: scale_point(end, scale),
                        },
                        PathElement::CurveTo {
                            control0,
                            control1,
                            end,
                        } => PathElement::CurveTo {
                            control0: scale_point(control0, scale),
                            control1: scale_point(control1, scale),
                            end: scale_point(end, scale),
                        },
                    })
                    .collect();
                Subpath::new(
                    scale_point(subpath.start(), scale),
                    elements,
                    subpath.is_closed(),
                )
            })
            .collect(),
    )
}

fn scale_point(point: Point, scale: f32) -> Point {
    Point::new(point.x * scale, point.y * scale)
}
