use skia_safe::{Path, PathBuilder, PathFillType};

use crate::geom::{Bounds, BoundsPen, Outline, PathElement};

pub(crate) mod remove_overlaps;

pub(crate) fn build_skia_path(
    outline: &Outline,
    track_bounds: bool,
    fill_type: PathFillType,
) -> Option<(Path, Option<Bounds>)> {
    let mut builder = PathBuilder::new_with_fill_type(fill_type);
    let mut bounds = track_bounds.then(BoundsPen::default);
    for subpath in outline.subpaths() {
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
